import os
import urllib.request
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                           miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

class AmazonReviewDownloader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # URLs for the Amazon review datasets
        self.urls = {
            'books': 'http://www.cs.jhu.edu/~mdredze/datasets/sentiment/processed_acl.tar.gz'
        }

    def download_and_extract(self):
        """Download and extract the Amazon review datasets"""
        for category, url in self.urls.items():
            # Download file
            compressed_file = self.raw_dir / f'{category}.tar.gz'
            if not compressed_file.exists():
                print(f'Downloading {category} reviews...')
                download_url(url, compressed_file)
            
            # Extract file
            extract_dir = self.raw_dir / category
            if not extract_dir.exists():
                print(f'Extracting {category} reviews...')
                shutil.unpack_archive(compressed_file, extract_dir)

    def get_review_files(self):
        """Get paths to positive and negative review files"""
        return {
            'positive': list(self.raw_dir.glob('**/positive.review')),
            'negative': list(self.raw_dir.glob('**/negative.review'))
        }

if __name__ == '__main__':
    downloader = AmazonReviewDownloader('data')
    downloader.download_and_extract()