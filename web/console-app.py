import torch
import sys
from pathlib import Path
import time
from tqdm import tqdm
import warnings

# Suppress the FutureWarning for torch.load
warnings.filterwarnings('ignore', category=FutureWarning)

# Add the src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
sys.path.append(str(src_path))

from model import SentimentAnalysisModel

class SentimentAnalyzerConsole:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load the trained model and preprocessor"""
        try:
            # Load preprocessor and data
            data_path = Path(__file__).parent.parent / 'data/processed/processed_data.pt'
            model_path = Path(__file__).parent.parent / 'models/best_model.pt'
            
            print("\nLoading model and preprocessor...")
            with tqdm(total=2) as pbar:
                # Load preprocessor
                try:
                    data = torch.load(data_path, weights_only=False, map_location=self.device)
                    self.preprocessor = data['preprocessor']
                    pbar.update(1)
                except Exception as e:
                    print(f"\n❌ Error loading preprocessor: {str(e)}")
                    return False
                
                # Initialize model
                self.model = SentimentAnalysisModel(
                    vocab_size=len(self.preprocessor.word2idx),
                    embedding_dim=100,
                    hidden_dim=256,
                    n_layers=2,
                    dropout=0.5
                ).to(self.device)
                
                # Load trained weights
                try:
                    checkpoint = torch.load(model_path, weights_only=True, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model.eval()
                    pbar.update(1)
                except Exception as e:
                    print(f"\n❌ Error loading model weights: {str(e)}")
                    return False
            
            print("✓ Model and preprocessor loaded successfully!")
            return True
            
        except Exception as e:
            print(f"\n❌ Error in model loading: {str(e)}")
            print("\nPlease ensure:")
            print(f"1. The processed data exists at: {data_path}")
            print(f"2. The model file exists at: {model_path}")
            return False

    def analyze_text(self, text):
        """Analyze sentiment of given text"""
        try:
            # Clean and preprocess the text
            tokens = self.preprocessor.clean_text(text)
            if not tokens:
                return None, "Invalid input text"
            
            # Encode the tokens
            encoded = self.preprocessor.encode_text(tokens)
            
            # Convert to tensor and add batch dimension
            tensor = torch.tensor([encoded]).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                prediction = self.model(tensor)
                confidence = float(prediction.item())
                
            sentiment = "Positive" if confidence >= 0.5 else "Negative"
            display_confidence = confidence if sentiment == "Positive" else 1 - confidence
            
            return {
                "sentiment": sentiment,
                "confidence": display_confidence,
                "processed_text": " ".join(tokens)
            }, None
            
        except Exception as e:
            return None, f"Error analyzing text: {str(e)}"

    def display_result(self, result):
        """Display analysis result in a formatted way"""
        sentiment = result["sentiment"]
        confidence = result["confidence"] * 100
        processed_text = result["processed_text"]
        
        # Create color codes
        GREEN = '\033[92m'
        RED = '\033[91m'
        BLUE = '\033[94m'
        BOLD = '\033[1m'
        END = '\033[0m'
        
        # Create sentiment color
        color = GREEN if sentiment == "Positive" else RED
        
        # Create confidence bar
        bar_length = 30
        filled_length = int(bar_length * confidence / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print("\n" + "="*60)
        print(f"{BOLD}Sentiment Analysis Results{END}")
        print("="*60)
        
        print(f"\n{BOLD}Sentiment:{END} {color}{sentiment}{END}")
        print(f"\n{BOLD}Confidence:{END}")
        print(f"{color}{bar}{END} {confidence:.1f}%")
        
        print(f"\n{BOLD}Processed Text:{END}")
        print(f"{BLUE}{processed_text}{END}")
        print("\n" + "="*60 + "\n")

    def run(self):
        """Run the console application"""
        print("\n" + "="*60)
        print("Welcome to Sentiment Analysis Console Application")
        print("="*60)
        
        if not self.load_model():
            print("Failed to load model. Exiting...")
            return
        
        while True:
            print("\nOptions:")
            print("1. Analyze text")
            print("2. Try sample texts")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                text = input("\nEnter the text to analyze (or 'back' to return to menu): ").strip()
                if text.lower() == 'back':
                    continue
                    
                print("\nAnalyzing...")
                result, error = self.analyze_text(text)
                if error:
                    print(f"\n❌ {error}")
                else:
                    self.display_result(result)
                    
            elif choice == '2':
                samples = [
                    "This product is absolutely amazing! The quality is exceptional.",
                    "Very disappointed with this purchase. Poor quality and terrible service.",
                    "Overall good product despite some minor issues. Worth the price."
                ]
                
                for i, sample in enumerate(samples, 1):
                    print(f"\nSample {i}:")
                    print(f"Text: {sample}")
                    result, error = self.analyze_text(sample)
                    if error:
                        print(f"❌ {error}")
                    else:
                        self.display_result(result)
                    time.sleep(1)  # Pause between samples
                    
            elif choice == '3':
                print("\nThank you for using Sentiment Analysis Console Application!")
                break
                
            else:
                print("\n❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    app = SentimentAnalyzerConsole()
    app.run()