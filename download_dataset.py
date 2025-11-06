import pandas as pd
import urllib.request
import zipfile
import os

def download_dataset():
    """Download and extract SMS Spam Collection dataset"""
    
    print("Downloading SMS Spam Collection dataset...")
    
    # Dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    
    # Download the zip file
    zip_path = "data/dataset.zip"
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Dataset downloaded successfully!")
        
        # Extract the zip file
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/")
        print("Dataset extracted successfully!")
        
        # Load and preview the data
        print("\nLoading dataset...")
        df = pd.read_csv('data/SMSSpamCollection', sep='\t', names=['label', 'text'])
        
        print(f"\n{'='*60}")
        print("DATASET INFORMATION")
        print(f"{'='*60}")
        print(f"Total emails: {len(df)}")
        print(f"Spam emails: {len(df[df['label'] == 'spam'])}")
        print(f"Ham emails: {len(df[df['label'] == 'ham'])}")
        print(f"{'='*60}")
        
        print("\nSample emails:")
        print("\n--- SPAM EXAMPLE ---")
        spam_sample = df[df['label'] == 'spam'].iloc[0]
        print(spam_sample['text'])
        
        print("\n--- HAM EXAMPLE ---")
        ham_sample = df[df['label'] == 'ham'].iloc[0]
        print(ham_sample['text'])
        
        # Save as CSV for easier processing
        df.to_csv('data/emails.csv', index=False)
        print("\nDataset saved as 'data/emails.csv'")
        
        # Clean up zip file
        os.remove(zip_path)
        print("Cleanup complete!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = download_dataset()
    if success:
        print("\nDataset setup complete!")
    else:
        print("\nDataset download failed. Please try again.")