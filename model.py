import joblib
import numpy as np
from preprocessing import TextPreprocessor
import os

class SpamClassifier:
    """
    Spam email classifier wrapper
    Loads trained model and makes predictions
    """
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.vectorizer = None
        self.load_model()
    
    def load_model(self):
        """
        Load trained model and vectorizer from disk
        """
        try:
            model_path = 'models/spam_classifier.pkl'
            vectorizer_path = 'models/tfidf_vectorizer.pkl'
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            if not os.path.exists(vectorizer_path):
                raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
            
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            
            print("Model and vectorizer loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict(self, text):
        """
        Predict if email is spam or ham
        
        Args:
            text (str): Email text to classify
            
        Returns:
            dict: Prediction results with confidence scores
        """
        if not text or not text.strip():
            raise ValueError("Email text cannot be empty")
        
        # Preprocess text
        processed_text = self.preprocessor.preprocess(text)
        
        # Vectorize
        text_tfidf = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(text_tfidf)[0]
        probability = self.model.predict_proba(text_tfidf)[0]
        
        # Get confidence score
        confidence = max(probability) * 100
        
        # Prepare result
        result = {
            'prediction': 'spam' if prediction == 1 else 'ham',
            'confidence': float(confidence),
            'spam_probability': float(probability[1] * 100),
            'ham_probability': float(probability[0] * 100),
            'is_spam': bool(prediction == 1)
        }
        
        return result
    
    def predict_batch(self, texts):
        """
        Predict multiple emails at once
        
        Args:
            texts (list): List of email texts to classify
            
        Returns:
            list: List of prediction results
        """
        results = []
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'prediction': None
                })
        
        return results
    
    def get_top_spam_features(self, n=20):
        """
        Get top N features that indicate spam
        
        Args:
            n (int): Number of top features to return
            
        Returns:
            list: Top spam indicator words
        """
        try:
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get feature log probabilities for spam class
            spam_class_index = 1
            feature_log_probs = self.model.feature_log_prob_[spam_class_index]
            
            # Get top N features
            top_indices = np.argsort(feature_log_probs)[-n:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            
            return top_features
            
        except Exception as e:
            print(f"Error getting top features: {e}")
            return []


# Test the classifier
if __name__ == "__main__":
    print("Testing Spam Classifier")
    print("="*60)
    
    # Initialize classifier
    classifier = SpamClassifier()
    
    # Test spam email
    spam_email = """
    WINNER! You have won $1,000,000 in our lottery!
    Click here NOW to claim your prize: http://scam.com
    This offer expires in 24 hours! Act fast!
    FREE MONEY! Limited time offer!
    """
    
    print("\nTest 1: Spam Email")
    print("-"*60)
    print("Text:", spam_email.strip()[:100] + "...")
    result = classifier.predict(spam_email)
    print(f"\nPrediction: {result['prediction'].upper()}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"Spam Probability: {result['spam_probability']:.2f}%")
    print(f"Ham Probability: {result['ham_probability']:.2f}%")
    
    # Test legitimate email
    ham_email = """
    Hi John,
    
    Just wanted to confirm our meeting tomorrow at 3 PM in conference room B.
    Please bring the quarterly reports for review.
    
    Thanks,
    Sarah
    """
    
    print("\n" + "="*60)
    print("\nTest 2: Legitimate Email")
    print("-"*60)
    print("Text:", ham_email.strip()[:100] + "...")
    result = classifier.predict(ham_email)
    print(f"\nPrediction: {result['prediction'].upper()}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"Spam Probability: {result['spam_probability']:.2f}%")
    print(f"Ham Probability: {result['ham_probability']:.2f}%")
    
    # Show top spam features
    print("\n" + "="*60)
    print("\nTop 20 Spam Indicator Words:")
    print("-"*60)
    top_features = classifier.get_top_spam_features(20)
    for i, feature in enumerate(top_features, 1):
        print(f"{i:2d}. {feature}")
    
    print("\n" + "="*60)
    print("Testing complete")