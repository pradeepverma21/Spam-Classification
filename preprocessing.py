import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
print("NLTK data downloaded successfully")

class TextPreprocessor:
    """
    Text preprocessing pipeline for email classification
    Includes: cleaning, tokenization, stopword removal, and stemming
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def clean_text(self, text):
        """
        Clean and normalize text
        - Convert to lowercase
        - Remove URLs, emails, punctuation, numbers
        - Remove extra whitespace
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words
        """
        return nltk.word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """
        Remove common stopwords (the, is, at, etc.)
        """
        return [word for word in tokens if word not in self.stop_words]
    
    def stem_words(self, tokens):
        """
        Apply stemming to reduce words to root form
        Example: running -> run, better -> better
        """
        return [self.stemmer.stem(word) for word in tokens]
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline
        Returns: cleaned and processed text as string
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Stem words
        tokens = self.stem_words(tokens)
        
        # Join back to string
        return ' '.join(tokens)


# Test the preprocessor
if __name__ == "__main__":
    print("\nTesting Text Preprocessor")
    print("="*60)
    
    preprocessor = TextPreprocessor()
    
    # Test with spam email
    spam_text = "WINNER!! You've won $1,000,000!!! Click here NOW: http://scam.com"
    print("\nOriginal Spam Text:")
    print(spam_text)
    print("\nProcessed Spam Text:")
    print(preprocessor.preprocess(spam_text))
    
    # Test with normal email
    ham_text = "Hey, are we still meeting for lunch tomorrow at 12pm?"
    print("\n" + "="*60)
    print("\nOriginal Ham Text:")
    print(ham_text)
    print("\nProcessed Ham Text:")
    print(preprocessor.preprocess(ham_text))
    
    print("\n" + "="*60)
    print("Preprocessing test complete")