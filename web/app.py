from flask import Flask, render_template, request, jsonify
import torch
import sys
from pathlib import Path

# Add the src directory to Python path so we can import our modules
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

from model import SentimentAnalysisModel

app = Flask(__name__)

# Global variables for model and preprocessor
model = None
preprocessor = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Load the trained model and preprocessor"""
    global model, preprocessor
    
    try:
        # Update paths to be relative to project root
        data_path = Path(__file__).parent.parent / 'data/processed/processed_data.pt'
        model_path = Path(__file__).parent.parent / 'models/best_model.pt'
        
        # Load preprocessor
        data = torch.load(data_path)
        preprocessor = data['preprocessor']
        
        # Initialize model
        model = SentimentAnalysisModel(
            vocab_size=len(preprocessor.word2idx),
            embedding_dim=100,
            hidden_dim=256,
            n_layers=2,
            dropout=0.5
        ).to(device)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("Model and preprocessor loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def predict_sentiment(text):
    """Predict sentiment for a given text"""
    # Clean and preprocess the text
    tokens = preprocessor.clean_text(text)
    if not tokens:
        return {"error": "Invalid input text"}
    
    # Encode the tokens
    encoded = preprocessor.encode_text(tokens)
    
    # Convert to tensor and add batch dimension
    tensor = torch.tensor([encoded]).to(device)
    
    # Get prediction
    with torch.no_grad():
        prediction = model(tensor)
        confidence = float(prediction.item())
        
    sentiment = "Positive" if confidence >= 0.5 else "Negative"
    
    return {
        "sentiment": sentiment,
        "confidence": confidence if sentiment == "Positive" else 1 - confidence,
        "processed_text": " ".join(tokens)
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "Please enter some text"}), 400
        
        result = predict_sentiment(text)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if load_model():
        app.run(debug=True)
    else:
        print("Failed to load model. Please ensure model files exist and paths are correct.")
        print("Required paths:")
        print(f"- Data: {Path(__file__).parent.parent / 'data/processed/processed_data.pt'}")
        print(f"- Model: {Path(__file__).parent.parent / 'models/best_model.pt'}")