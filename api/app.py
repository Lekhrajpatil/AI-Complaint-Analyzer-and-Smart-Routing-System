"""
Production-ready Flask API for AI Complaint Analyzer
Optimized for Vercel serverless deployment
"""

from flask import Flask, request, jsonify, render_template
import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
           template_folder='../templates',
           static_folder='../static')

# Global variables for models
models_loaded = False
category_model = None
priority_model = None
vectorizer = None
category_encoder = None
priority_encoder = None
lemmatizer = None
stop_words = None

# Department mapping
DEPARTMENT_MAPPING = {
    'Billing': 'Finance Department',
    'Technical Issue': 'Technical Support',
    'Service': 'Customer Service',
    'Product': 'Product Team',
    'Delivery': 'Logistics Department',
    'Others': 'General Support'
}

def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        logger.warning(f"NLTK download failed: {e}")

def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        text = str(text)
    
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

def tokenize_and_lemmatize(text):
    """Tokenize and lemmatize text"""
    try:
        tokens = word_tokenize(text)
        lemmatized_tokens = []
        
        for token in tokens:
            if token not in stop_words and len(token) > 2:
                lemmatized_token = lemmatizer.lemmatize(token)
                lemmatized_tokens.append(lemmatized_token)
        
        return ' '.join(lemmatized_tokens)
    except:
        return text

def preprocess_text(text):
    """Complete text preprocessing pipeline"""
    try:
        cleaned_text = clean_text(text)
        processed_text = tokenize_and_lemmatize(cleaned_text)
        return processed_text
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        return text

def load_models():
    """Load pre-trained models"""
    global models_loaded, category_model, priority_model, vectorizer
    global category_encoder, priority_encoder, lemmatizer, stop_words
    
    try:
        # Download NLTK data
        download_nltk_data()
        
        # Initialize NLP components
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        # Load models from pickle files
        model_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load complete preprocessor
        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                preprocessor_data = pickle.load(f)
                vectorizer = preprocessor_data['tfidf_vectorizer']
                category_encoder = preprocessor_data['category_encoder']
                priority_encoder = preprocessor_data['priority_encoder']
        else:
            logger.error("Preprocessor file not found")
            return False
        
        # Load category model
        category_model_path = os.path.join(model_dir, 'category_model.pkl')
        if os.path.exists(category_model_path):
            with open(category_model_path, 'rb') as f:
                category_model = pickle.load(f)
        else:
            logger.error("Category model file not found")
            return False
        
        # Load priority model
        priority_model_path = os.path.join(model_dir, 'priority_model.pkl')
        if os.path.exists(priority_model_path):
            with open(priority_model_path, 'rb') as f:
                priority_model = pickle.load(f)
        else:
            logger.error("Priority model file not found")
            return False
        
        models_loaded = True
        logger.info("Models loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

def validate_complaint_text(text):
    """Validate complaint text input"""
    if not text or not text.strip():
        return False, "Complaint text cannot be empty"
    
    if len(text.strip()) < 10:
        return False, "Complaint text must be at least 10 characters long"
    
    if len(text) > 5000:
        return False, "Complaint text must be less than 5000 characters"
    
    return True, ""

def get_priority_color(priority):
    """Get color code for priority level"""
    colors = {
        'High': '#dc3545',    # Red
        'Medium': '#ffc107',  # Yellow
        'Low': '#28a745'      # Green
    }
    return colors.get(priority, '#6c757d')

def get_category_icon(category):
    """Get icon for category"""
    icons = {
        'Billing': '💳',
        'Technical Issue': '🔧',
        'Service': '👥',
        'Product': '📦',
        'Delivery': '🚚',
        'Others': '📋'
    }
    return icons.get(category, '📋')

def predict_complaint(complaint_text):
    """Make prediction for complaint text"""
    try:
        if not models_loaded:
            return None, "Models not loaded"
        
        # Validate input
        is_valid, error_msg = validate_complaint_text(complaint_text)
        if not is_valid:
            return None, error_msg
        
        # Preprocess text
        processed_text = preprocess_text(complaint_text)
        
        # Extract features
        features = vectorizer.transform([processed_text])
        
        # Make predictions
        category_pred = category_model.predict(features)[0]
        priority_pred = priority_model.predict(features)[0]
        
        # Get probabilities
        category_proba = category_model.predict_proba(features)[0]
        priority_proba = priority_model.predict_proba(features)[0]
        
        # Convert labels back to original
        category_label = category_encoder.inverse_transform([category_pred])[0]
        priority_label = priority_encoder.inverse_transform([priority_pred])[0]
        
        # Get department
        department = DEPARTMENT_MAPPING.get(category_label, 'General Support')
        
        result = {
            'category': category_label,
            'priority': priority_label,
            'department': department,
            'category_confidence': float(max(category_proba)),
            'priority_confidence': float(max(priority_proba)),
            'category_icon': get_category_icon(category_label),
            'priority_color': get_priority_color(priority_label)
        }
        
        return result, None
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return None, str(e)

# Load models on startup
load_models()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for complaint prediction"""
    try:
        # Get request data
        if request.is_json:
            data = request.get_json()
            complaint_text = data.get('complaint_text', '')
        else:
            complaint_text = request.form.get('complaint_text', '')
        
        # Make prediction
        result, error = predict_complaint(complaint_text)
        
        if error:
            if request.is_json:
                return jsonify({
                    'error': error,
                    'status': 'error'
                }), 400
            else:
                return render_template('index.html', error=error)
        
        if request.is_json:
            return jsonify({
                'status': 'success',
                'predictions': result
            })
        else:
            return render_template('index.html', result=result, success=True)
            
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        if request.is_json:
            return jsonify({
                'error': 'An error occurred during prediction',
                'status': 'error'
            }), 500
        else:
            return render_template('index.html', 
                                 error='An error occurred during prediction')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded
    })

@app.route('/model_info')
def model_info():
    """Get model information"""
    if not models_loaded:
        return jsonify({
            'error': 'Models not loaded',
            'status': 'error'
        }), 500
    
    try:
        info = {
            'status': 'success',
            'categories': category_encoder.classes_.tolist(),
            'priorities': priority_encoder.classes_.tolist(),
            'department_mapping': DEPARTMENT_MAPPING
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({
            'error': 'Error retrieving model information',
            'status': 'error'
        }), 500

# Vercel serverless handler
def handler(environ, start_response):
    """Vercel serverless function handler"""
    return app(environ, start_response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
