import nltk

def download_nltk_data():
    """Download all required NLTK data"""
    resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'omw-1.4',
        'averaged_perceptron_tagger'
    ]
    
    for resource in resources:
        print(f"Downloading {resource}...")
        nltk.download(resource)

if __name__ == '__main__':
    download_nltk_data()