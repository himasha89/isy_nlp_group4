# Sentiment Analysis System
ISY503 Assessment 3 - Group 4

## Overview
A comprehensive sentiment analysis system that analyzes Amazon product reviews using Natural Language Processing (NLP). The system features both web and console interfaces for real-time sentiment analysis, providing sentiment classification with confidence scores.

## Team Members
- Himasha - Core ML Implementation & Leadership
- Carlos - Technical Optimization & Ethics
- Aashish - Data Processing Pipeline
- Sohail - Interface Development
- Simon - Documentation & Support

## Features
- Real-time sentiment analysis
- Web and console interfaces
- Text preprocessing pipeline
- Confidence score visualization
- Ethical bias detection and mitigation
- Comprehensive documentation

## Directory Structure
```
ISY_NLP_GROUP4/
├── data/                    # Data directory
├── models/                  # Saved model checkpoints
├── src/
│   ├── _pycache_/          # Python cache files
│   ├── config.py           # Configuration settings
│   ├── data_downloader.py  # Data download utilities
│   ├── data_preprocessing.py# Data preprocessing pipeline
│   ├── model.py            # Model architecture
│   ├── run_preprocessing.py# Preprocessing execution script
│   ├── run_training.py     # Training execution script
│   ├── setup_nltk.py       # NLTK setup script
│   └── train.py            # Training pipeline
├── web/
│   ├── static/             # Static files
│   ├── templates/          # HTML templates
│   ├── app.py              # Flask web application
│   └── console_app.py      # Console interface
├── .gitignore              # Git ignore file
├── README.md               # Project documentation
└── requirements.txt        # Project dependencies
```

## Prerequisites
- Python 3.11 or higher
- pip package manager
- Virtual environment (recommended)
- Git for version control

## Installation

1. Clone the repository
```bash
git clone [repository-url]
cd ISY_NLP_GROUP4
```

2. Create and activate virtual environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download NLTK Data
```bash
python src/setup_nltk.py
```

## Usage

### Data Preprocessing
```bash
python src/run_preprocessing.py
```

### Model Training
```bash
python src/run_training.py
```

### Running Web Interface
```bash
python web/app.py
```
Access the web interface at: http://localhost:5000

### Running Console Interface
```bash
python web/console_app.py
```

## Example Usage

### Web Interface
1. Navigate to http://localhost:5000
2. Enter text in input field
3. Click "Analyze" button
4. View sentiment analysis results

### Console Interface
```bash
# Example positive review
"This product is absolutely amazing! The quality is exceptional and it exceeded all my expectations."

# Example negative review
"Very disappointed with this purchase. Poor quality and terrible customer service."
```

## Technical Details

### Model Architecture
- Bidirectional LSTM with attention mechanism
- Word embeddings (100 dimensions)
- Hidden layers: 256 units
- Dropout rate: 0.5

### Data Processing
- Text cleaning and normalization
- Tokenization using NLTK
- Feature extraction
- Quality assurance checks

### Ethical Considerations
- Bias detection and mitigation
- Fairness metrics implementation
- Transparency measures
- Model confidence scoring

## Troubleshooting

### Common Issues
1. ModuleNotFoundError
   - Ensure virtual environment is activated
   - Verify all dependencies are installed

2. Model Loading Error
   - Check model file exists in models/best_model.pt
   - Verify processed data exists in data/processed/processed_data.pt

3. NLTK Data Error
   - Run setup_nltk.py to download required data

## Team Contributions
- Core ML Implementation (Himasha): 30%
- Technical Optimization & Ethics (Carlos): 20%
- Data Processing Pipeline (Aashish): 17%
- Interface Development (Sohail): 17%
- Documentation & Support (Simon): 16%

## Acknowledgments
- Dataset: Amazon product reviews
- NLTK for text processing
- PyTorch for deep learning
- Flask for web interface
