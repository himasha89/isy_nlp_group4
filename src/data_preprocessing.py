import re
import nltk
import torch
import numpy as np
from pathlib import Path
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class ReviewPreprocessor:
    def __init__(self, max_length=100, min_length=5, min_word_freq=2):
        self.max_length = max_length
        self.min_length = min_length
        self.min_word_freq = min_word_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = Counter()
        
        # Initialize NLTK components
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            nltk.download('averaged_perceptron_tagger')
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()

    def tokenize_text(self, text):
        """Simple tokenization by splitting on whitespace"""
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Split on whitespace
        return text.lower().split()

    def clean_text(self, text):
        """Clean and normalize the text"""
        try:
            # Convert to lowercase and basic cleaning
            text = text.lower().strip()
            
            # Simple tokenization
            tokens = self.tokenize_text(text)
            
            # Remove stopwords and lemmatize
            tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token not in self.stop_words and len(token) > 1
            ]
            
            return tokens
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return []

    def build_vocabulary(self, token_lists):
        """Build vocabulary from list of token lists"""
        print("Building word frequency counter...")
        # Reset word frequency counter
        self.word_freq = Counter()
        
        # Count frequencies
        for tokens in token_lists:
            self.word_freq.update(tokens)
        
        print(f"Total unique words before filtering: {len(self.word_freq)}")
        
        # Reset word2idx with special tokens
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        
        # Add words that appear more than min_word_freq times
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_word_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        print(f"Vocabulary size after filtering: {len(self.word2idx)}")
        
        return self.word2idx

    def encode_text(self, tokens):
        """Convert token list to sequence of indices"""
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + ['<PAD>'] * (self.max_length - len(tokens))
        
        # Convert to indices
        return [self.word2idx.get(token, self.word2idx['<UNK>']) 
                for token in tokens]

    def decode_text(self, indices):
        """Convert sequence of indices back to tokens"""
        return [self.idx2word[idx] for idx in indices]

    @property
    def vocab_size(self):
        """Return the size of the vocabulary"""
        return len(self.word2idx)

class ReviewDataset(Dataset):
    def __init__(self, reviews, labels):
        self.reviews = torch.tensor(reviews, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.reviews[idx], self.labels[idx]

class DataProcessor:
    def __init__(self, data_dir, preprocessor):
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor

    def read_reviews(self, file_path):
        """Read reviews from a file with error handling"""
        reviews = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Skip empty lines
                    if line.strip():
                        reviews.append(line.strip())
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
        return reviews

    def process_data(self):
        """Process all reviews and prepare for training"""
        # Read positive and negative reviews
        pos_files = list(self.data_dir.glob('**/positive.review'))
        neg_files = list(self.data_dir.glob('**/negative.review'))
        
        if not pos_files or not neg_files:
            raise FileNotFoundError("Review files not found. Check the data directory.")
        
        print("Reading positive reviews...")
        positive_reviews = []
        for file in pos_files:
            positive_reviews.extend(self.read_reviews(file))
            
        print("Reading negative reviews...")
        negative_reviews = []
        for file in neg_files:
            negative_reviews.extend(self.read_reviews(file))
        
        print(f"Found {len(positive_reviews)} positive and {len(negative_reviews)} negative reviews")
        
        # Clean texts
        print("Cleaning texts...")
        pos_tokens = [self.preprocessor.clean_text(text) for text in positive_reviews]
        neg_tokens = [self.preprocessor.clean_text(text) for text in negative_reviews]
        
        # Remove empty reviews
        pos_tokens = [tokens for tokens in pos_tokens if len(tokens) >= self.preprocessor.min_length]
        neg_tokens = [tokens for tokens in neg_tokens if len(tokens) >= self.preprocessor.min_length]
        
        print(f"After cleaning: {len(pos_tokens)} positive and {len(neg_tokens)} negative reviews")
        
        # Build vocabulary
        print("Building vocabulary...")
        self.preprocessor.build_vocabulary(pos_tokens + neg_tokens)
        
        # Encode reviews
        print("Encoding reviews...")
        pos_encoded = [self.preprocessor.encode_text(tokens) for tokens in pos_tokens]
        neg_encoded = [self.preprocessor.encode_text(tokens) for tokens in neg_tokens]
        
        # Create labels
        pos_labels = [1] * len(pos_encoded)
        neg_labels = [0] * len(neg_encoded)
        
        # Combine data
        X = np.array(pos_encoded + neg_encoded)
        y = np.array(pos_labels + neg_labels)
        
        # Split data
        print("Splitting data into train/val/test sets...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def create_data_loaders(train_data, val_data, test_data, batch_size=32):
    """Create PyTorch DataLoaders"""
    train_dataset = ReviewDataset(train_data[0], train_data[1])
    val_dataset = ReviewDataset(val_data[0], val_data[1])
    test_dataset = ReviewDataset(test_data[0], test_data[1])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader