from data_downloader import AmazonReviewDownloader
from data_preprocessing import ReviewPreprocessor, DataProcessor, create_data_loaders
import torch
from pathlib import Path

def main():
    # Setup paths
    data_dir = Path('data')
    
    # Download data
    print("Downloading dataset...")
    downloader = AmazonReviewDownloader(data_dir)
    downloader.download_and_extract()
    
    # Initialize preprocessor
    print("Initializing preprocessor...")
    preprocessor = ReviewPreprocessor(
        max_length=100,
        min_length=5,
        min_word_freq=2
    )
    
    # Process data
    print("Processing data...")
    processor = DataProcessor(data_dir / 'raw', preprocessor)
    train_data, val_data, test_data = processor.process_data()
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, batch_size=32
    )
    
    # Save preprocessed data and preprocessor
    print("Saving processed data...")
    torch.save({
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'preprocessor': preprocessor
    }, data_dir / 'processed' / 'processed_data.pt')
    
    print("Data preprocessing completed!")
    print(f"Vocabulary size: {len(preprocessor.word2idx)}")
    print(f"Training samples: {len(train_data[0])}")
    print(f"Validation samples: {len(val_data[0])}")
    print(f"Test samples: {len(test_data[0])}")

if __name__ == '__main__':
    main()