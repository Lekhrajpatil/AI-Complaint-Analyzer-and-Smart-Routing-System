"""
Text Preprocessing Module for Complaint Analysis
Handles NLP preprocessing including cleaning, tokenization, and feature extraction
"""

import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import logging
from typing import Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComplaintPreprocessor:
    """Handles all text preprocessing for complaint analysis"""
    
    def __init__(self):
        """Initialize the preprocessor with necessary NLP tools"""
        try:
            # Download required NLTK data
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            
            # Initialize NLP components
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            # Initialize encoders
            self.category_encoder = LabelEncoder()
            self.priority_encoder = LabelEncoder()
            
            # Initialize TF-IDF Vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                stop_words='english'
            )
            
            # Store fitted status
            self.is_fitted = False
            
            logger.info("Preprocessor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing preprocessor: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            text = str(text)
        
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove phone numbers
            text = re.sub(r'\d{3}-\d{3}-\d{4}|\d{10}', '', text)
            
            # Remove currency symbols and amounts
            text = re.sub(r'\$\d+\.?\d*', '', text)
            
            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            
            # Remove numbers
            text = re.sub(r'\d+', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text
    
    def tokenize_and_lemmatize(self, text: str) -> str:
        """
        Tokenize text and apply lemmatization
        
        Args:
            text: Cleaned text string
            
        Returns:
            Lemmatized text string
        """
        try:
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            lemmatized_tokens = []
            for token in tokens:
                if token not in self.stop_words and len(token) > 2:
                    lemmatized_token = self.lemmatizer.lemmatize(token)
                    lemmatized_tokens.append(lemmatized_token)
            
            return ' '.join(lemmatized_tokens)
            
        except Exception as e:
            logger.error(f"Error in tokenization and lemmatization: {str(e)}")
            return text
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete text preprocessing pipeline
        
        Args:
            text: Raw text string
            
        Returns:
            Fully preprocessed text
        """
        try:
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Tokenize and lemmatize
            processed_text = self.tokenize_and_lemmatize(cleaned_text)
            
            return processed_text
            
        except Exception as e:
            logger.error(f"Error in text preprocessing: {str(e)}")
            return text
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit preprocessing pipeline and transform data
        
        Args:
            df: DataFrame with complaint data
            
        Returns:
            Tuple of (X_features, y_category, y_priority)
        """
        try:
            logger.info("Starting preprocessing pipeline...")
            
            # Preprocess complaint texts
            logger.info("Preprocessing complaint texts...")
            df['processed_text'] = df['complaint_text'].apply(self.preprocess_text)
            
            # Remove empty texts after preprocessing
            df = df[df['processed_text'].str.len() > 0]
            logger.info(f"Removed {len(df) - len(df)} empty texts")
            
            # Extract features using TF-IDF
            logger.info("Extracting TF-IDF features...")
            X_features = self.tfidf_vectorizer.fit_transform(df['processed_text'])
            
            # Encode labels
            logger.info("Encoding labels...")
            y_category = self.category_encoder.fit_transform(df['category'])
            y_priority = self.priority_encoder.fit_transform(df['priority'])
            
            # Mark as fitted
            self.is_fitted = True
            
            logger.info(f"Preprocessing completed. Feature shape: {X_features.shape}")
            logger.info(f"Categories: {list(self.category_encoder.classes_)}")
            logger.info(f"Priorities: {list(self.priority_encoder.classes_)}")
            
            return X_features, y_category, y_priority
            
        except Exception as e:
            logger.error(f"Error in fit_transform: {str(e)}")
            raise
    
    def transform(self, texts: list) -> np.ndarray:
        """
        Transform new texts using fitted pipeline
        
        Args:
            texts: List of text strings to transform
            
        Returns:
            Transformed features
        """
        try:
            if not self.is_fitted:
                raise ValueError("Preprocessor must be fitted before transform")
            
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Transform using fitted TF-IDF
            features = self.tfidf_vectorizer.transform(processed_texts)
            
            return features
            
        except Exception as e:
            logger.error(f"Error in transform: {str(e)}")
            raise
    
    def inverse_transform_category(self, encoded_labels: np.ndarray) -> list:
        """Convert encoded category labels back to original labels"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        return self.category_encoder.inverse_transform(encoded_labels)
    
    def inverse_transform_priority(self, encoded_labels: np.ndarray) -> list:
        """Convert encoded priority labels back to original labels"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        return self.priority_encoder.inverse_transform(encoded_labels)
    
    def get_feature_names(self) -> list:
        """Get feature names from TF-IDF vectorizer"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        return self.tfidf_vectorizer.get_feature_names_out().tolist()
    
    def save_preprocessor(self, filepath: str):
        """
        Save the fitted preprocessor to file
        
        Args:
            filepath: Path to save the preprocessor
        """
        try:
            if not self.is_fitted:
                raise ValueError("Preprocessor must be fitted before saving")
            
            preprocessor_data = {
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'category_encoder': self.category_encoder,
                'priority_encoder': self.priority_encoder,
                'is_fitted': self.is_fitted
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(preprocessor_data, f)
            
            logger.info(f"Preprocessor saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving preprocessor: {str(e)}")
            raise
    
    def load_preprocessor(self, filepath: str):
        """
        Load fitted preprocessor from file
        
        Args:
            filepath: Path to load the preprocessor from
        """
        try:
            with open(filepath, 'rb') as f:
                preprocessor_data = pickle.load(f)
            
            self.tfidf_vectorizer = preprocessor_data['tfidf_vectorizer']
            self.category_encoder = preprocessor_data['category_encoder']
            self.priority_encoder = preprocessor_data['priority_encoder']
            self.is_fitted = preprocessor_data['is_fitted']
            
            logger.info(f"Preprocessor loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading preprocessor: {str(e)}")
            raise
    
    def get_preprocessing_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about preprocessing
        
        Args:
            df: Original DataFrame
            
        Returns:
            Dictionary with preprocessing statistics
        """
        try:
            stats = {
                'total_samples': len(df),
                'categories': df['category'].value_counts().to_dict(),
                'priorities': df['priority'].value_counts().to_dict(),
                'avg_text_length': df['complaint_text'].str.len().mean(),
                'avg_processed_length': 0
            }
            
            if 'processed_text' in df.columns:
                stats['avg_processed_length'] = df['processed_text'].str.len().mean()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting preprocessing stats: {str(e)}")
            return {}

def main():
    """Test the preprocessor"""
    try:
        # Sample data for testing
        sample_data = {
            'complaint_text': [
                "I was charged $99.99 for a service I didn't subscribe to!",
                "The mobile app keeps crashing when I try to upload files",
                "Customer service was very rude and unhelpful"
            ],
            'category': ['Billing', 'Technical Issue', 'Service'],
            'priority': ['High', 'Medium', 'Low']
        }
        
        df = pd.DataFrame(sample_data)
        
        # Initialize and test preprocessor
        preprocessor = ComplaintPreprocessor()
        X, y_cat, y_pri = preprocessor.fit_transform(df)
        
        logger.info("Preprocessor test completed successfully!")
        logger.info(f"Feature shape: {X.shape}")
        logger.info(f"Sample features: {X[0].toarray()[:5]}")
        
    except Exception as e:
        logger.error(f"Error in preprocessor test: {str(e)}")

if __name__ == "__main__":
    main()
