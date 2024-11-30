import torch
from pathlib import Path
from model import SentimentAnalysisModel
from train import ModelTrainer
from data_preprocessing import create_data_loaders

def main():
    # Load preprocessed data
    print("Loading preprocessed data...")
    data = torch.load('data/processed/processed_data.pt')
    
    # Create data loaders with smaller batch size for better generalization
    train_loader, val_loader, test_loader = create_data_loaders(
        data['train_data'],
        data['val_data'],
        data['test_data'],
        batch_size=16  # Reduced batch size
    )
    
    # Initialize model with adjusted parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SentimentAnalysisModel(
        vocab_size=len(data['preprocessor'].word2idx),
        embedding_dim=200,    # Increased embedding dimension
        hidden_dim=512,       # Increased hidden dimension
        n_layers=3,          # Increased number of layers
        dropout=0.3          # Adjusted dropout
    )
    
    # Initialize trainer with adjusted learning rate
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=0.0003,  # Reduced learning rate for better convergence
        save_dir='models'
    )
    
    # Train model with more epochs and patience
    print("\nStarting training...")
    trainer.train(
        num_epochs=20,           # Increased number of epochs
        early_stopping_patience=5 # Increased patience
    )

if __name__ == '__main__':
    main()