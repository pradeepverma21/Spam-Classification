import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
from preprocessing import TextPreprocessor
import matplotlib.pyplot as plt
import seaborn as sns
import os

class SpamClassifierTrainer:
    """
    Train a Naive Bayes spam classifier using TF-IDF features
    """
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=3000, min_df=2, max_df=0.8)
        self.model = MultinomialNB(alpha=0.1)
        self.metrics = {}
        
    def load_data(self, filepath):
        """
        Load dataset from CSV file
        """
        print("Loading dataset...")
        df = pd.read_csv(filepath)
        
        # Convert labels to binary (spam=1, ham=0)
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
        
        print(f"Dataset loaded: {len(df)} emails")
        print(f"Spam: {sum(df['label'] == 1)} ({sum(df['label'] == 1)/len(df)*100:.2f}%)")
        print(f"Ham: {sum(df['label'] == 0)} ({sum(df['label'] == 0)/len(df)*100:.2f}%)")
        
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess all text data
        """
        print("\nPreprocessing texts...")
        df['processed_text'] = df['text'].apply(self.preprocessor.preprocess)
        print("Preprocessing complete")
        
        # Show examples
        print("\nExample preprocessed texts:")
        print("-" * 60)
        spam_example = df[df['label'] == 1].iloc[0]
        print(f"SPAM - Original: {spam_example['text'][:80]}...")
        print(f"SPAM - Processed: {spam_example['processed_text'][:80]}...")
        print()
        ham_example = df[df['label'] == 0].iloc[0]
        print(f"HAM - Original: {ham_example['text'][:80]}...")
        print(f"HAM - Processed: {ham_example['processed_text'][:80]}...")
        print("-" * 60)
        
        return df
    
    def prepare_features(self, X_train, X_test):
        """
        Convert text to TF-IDF features
        """
        print("\nCreating TF-IDF features...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"Feature matrix shape: {X_train_tfidf.shape}")
        print(f"Number of features: {len(self.vectorizer.get_feature_names_out())}")
        
        return X_train_tfidf, X_test_tfidf
    
    def train(self, X_train, y_train):
        """
        Train the Naive Bayes model
        """
        print("\nTraining Naive Bayes model...")
        self.model.fit(X_train, y_train)
        print("Training complete")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        """
        print("\nEvaluating model...")
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred) * 100,
            'precision': precision_score(y_test, y_pred) * 100,
            'recall': recall_score(y_test, y_pred) * 100,
            'f1_score': f1_score(y_test, y_pred) * 100,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return self.metrics
    
    def print_metrics(self):
        """
        Print detailed evaluation metrics
        """
        print("\n" + "="*60)
        print("MODEL PERFORMANCE METRICS")
        print("="*60)
        print(f"Accuracy:  {self.metrics['accuracy']:.2f}%")
        print(f"Precision: {self.metrics['precision']:.2f}%")
        print(f"Recall:    {self.metrics['recall']:.2f}%")
        print(f"F1-Score:  {self.metrics['f1_score']:.2f}%")
        print("\nConfusion Matrix:")
        print("                Predicted")
        print("              Ham    Spam")
        cm = self.metrics['confusion_matrix']
        print(f"Actual Ham   {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"Actual Spam  {cm[1][0]:5d}  {cm[1][1]:5d}")
        print("="*60)
    
    def plot_confusion_matrix(self):
        """
        Plot and save confusion matrix
        """
        try:
            plt.figure(figsize=(8, 6))
            cm = self.metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Ham', 'Spam'], 
                       yticklabels=['Ham', 'Spam'])
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            plt.savefig('models/confusion_matrix.png')
            print("\nConfusion matrix plot saved to models/confusion_matrix.png")
            plt.close()
        except Exception as e:
            print(f"Could not save confusion matrix plot: {e}")
    
    def save_model(self):
        """
        Save trained model and vectorizer
        """
        print("\nSaving model and vectorizer...")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model and vectorizer
        joblib.dump(self.model, 'models/spam_classifier.pkl')
        joblib.dump(self.vectorizer, 'models/tfidf_vectorizer.pkl')
        
        # Save metrics
        import json
        with open('models/metrics.json', 'w') as f:
            metrics_to_save = {
                'accuracy': self.metrics['accuracy'],
                'precision': self.metrics['precision'],
                'recall': self.metrics['recall'],
                'f1_score': self.metrics['f1_score']
            }
            json.dump(metrics_to_save, f, indent=4)
        
        print("Model saved to models/spam_classifier.pkl")
        print("Vectorizer saved to models/tfidf_vectorizer.pkl")
        print("Metrics saved to models/metrics.json")
    
    def run_training_pipeline(self, filepath):
        """
        Complete training pipeline
        """
        print("="*60)
        print("SPAM EMAIL CLASSIFIER TRAINING")
        print("="*60)
        
        # Load data
        df = self.load_data(filepath)
        
        # Preprocess
        df = self.preprocess_data(df)
        
        # Split data (80% train, 20% test)
        print("\nSplitting data (80% train, 20% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df['label'], 
            test_size=0.2, 
            random_state=42,
            stratify=df['label']
        )
        print(f"Training set: {len(X_train)} emails")
        print(f"Testing set: {len(X_test)} emails")
        
        # Prepare features
        X_train_tfidf, X_test_tfidf = self.prepare_features(X_train, X_test)
        
        # Train
        self.train(X_train_tfidf, y_train)
        
        # Evaluate
        self.evaluate(X_test_tfidf, y_test)
        self.print_metrics()
        self.plot_confusion_matrix()
        
        # Save
        self.save_model()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)


if __name__ == "__main__":
    trainer = SpamClassifierTrainer()
    trainer.run_training_pipeline('data/emails.csv')