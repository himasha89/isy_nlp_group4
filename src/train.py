import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

class ModelTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        learning_rate=0.001,
        save_dir='models'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize loss function and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, verbose=True)
        
        # Initialize tracking variables
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        
        # Training info
        self.training_info = {
            'model_params': model.count_parameters(),
            'optimizer': type(self.optimizer).__name__,
            'learning_rate': learning_rate,
            'device': str(device),
            'started': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def train_epoch(self):
        """Train the model for one epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch in progress_bar:
            text, labels = [item.to(self.device) for item in batch]
            
            self.optimizer.zero_grad()
            predictions = self.model(text)
            loss = self.criterion(predictions, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            predicted_labels = (predictions >= 0.5).long()
            correct_predictions += (predicted_labels == labels.long()).sum().item()
            total_predictions += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct_predictions/total_predictions:.4f}'
            })
        
        return total_loss / len(self.train_loader), correct_predictions / total_predictions

    def evaluate(self, data_loader, desc="Evaluating"):
        """Evaluate the model on the given data loader"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc=desc)
            for batch in progress_bar:
                text, labels = [item.to(self.device) for item in batch]
                
                predictions = self.model(text)
                loss = self.criterion(predictions, labels)
                
                total_loss += loss.item()
                predicted_labels = (predictions >= 0.5).long()
                correct_predictions += (predicted_labels == labels.long()).sum().item()
                total_predictions += labels.size(0)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct_predictions/total_predictions:.4f}'
                })
        
        return total_loss / len(data_loader), correct_predictions / total_predictions

    def train(self, num_epochs, early_stopping_patience=5):
        """Train the model for the specified number of epochs"""
        print(f"Starting training with {self.training_info['model_params']:,} parameters")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation phase
            val_loss, val_acc = self.evaluate(self.val_loader, "Validating")
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f'\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
            print(f'Val. Loss: {val_loss:.4f} |  Val. Acc: {val_acc*100:.2f}%')
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model('best_model.pt', epoch, val_loss, val_acc)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break
            
        # Final evaluation on test set
        test_loss, test_acc = self.evaluate(self.test_loader, "Testing")
        print(f'\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%')
        
        # Save training history
        self.save_training_history()
        
        return test_loss, test_acc

    def save_model(self, filename, epoch, val_loss, val_acc):
        """Save model checkpoint"""
        path = self.save_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'training_info': self.training_info
        }, path)
        print(f'Model saved to {path}')

    def save_training_history(self):
        """Save training history and plots"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'training_info': self.training_info
        }
        
        # Save history as JSON
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump({k: v if not isinstance(v, list) else [float(i) for i in v] 
                      for k, v in history.items()}, f, indent=4)
        
        # Plot training history
        self.plot_training_history()

    def plot_training_history(self):
        """Plot and save training history graphs"""
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_history.png')
        plt.close()