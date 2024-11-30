import torch
from pathlib import Path
from model import SentimentAnalysisModel
from train import ModelTrainer
from data_preprocessing import create_data_loaders

def main():
    # Load preprocessed data
    print("Loading preprocessed data...")
    data = torch.load('data/processed/processed_data.pt')
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data['train_data'],
        data['val_data'],
        data['test_data'],
        batch_size=32
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SentimentAnalysisModel(
        vocab_size=len(data['preprocessor'].word2idx),
        embedding_dim=100,
        hidden_dim=256,
        n_layers=2,
        dropout=0.5
    )
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=0.001,
        save_dir='models'
    )
    
    # Train model
    print("\nStarting training...")
    trainer.train(num_epochs=10, early_stopping_patience=3)

if __name__ == '__main__':
    main()