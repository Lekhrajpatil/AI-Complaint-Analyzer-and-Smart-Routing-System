"""
Flask Web Application for AI Complaint Analyzer and Smart Routing System
Provides web interface and REST API for complaint classification
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import logging
import os
from datetime import datetime
import json
from train_model import ComplaintModelTrainer
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'complaint_analyzer_secret_key_2024'

# Global variables for models
model_trainer = None
models_loaded = False

def load_models():
    """Load trained models on application startup"""
    global model_trainer, models_loaded
    
    try:
        logger.info("Loading trained models...")
        
        # Initialize trainer
        model_trainer = ComplaintModelTrainer()
        
        # Check if model files exist
        model_files = [
            'models/preprocessor.pkl',
            'models/category_model.pkl', 
            'models/priority_model.pkl'
        ]
        
        for file_path in model_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Load models
        model_trainer.load_models(
            'models/preprocessor.pkl',
            'models/category_model.pkl',
            'models/priority_model.pkl'
        )
        
        models_loaded = True
        logger.info("Models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        models_loaded = False

def validate_complaint_text(text: str) -> tuple:
    """
    Validate complaint text input
    
    Args:
        text: Complaint text to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Complaint text cannot be empty"
    
    if len(text.strip()) < 10:
        return False, "Complaint text must be at least 10 characters long"
    
    if len(text) > 5000:
        return False, "Complaint text must be less than 5000 characters"
    
    return True, ""

def get_priority_color(priority: str) -> str:
    """Get color code for priority level"""
    colors = {
        'High': '#dc3545',    # Red
        'Medium': '#ffc107',  # Yellow
        'Low': '#28a745'      # Green
    }
    return colors.get(priority, '#6c757d')

def get_category_icon(category: str) -> str:
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

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    REST API endpoint for complaint prediction
    Accepts JSON: {"complaint_text": "your complaint here"}
    Returns JSON with predictions
    """
    try:
        # Check if models are loaded
        if not models_loaded:
            return jsonify({
                'error': 'Models not loaded. Please train the models first.',
                'status': 'error'
            }), 500
        
        # Get request data
        data = request.get_json()
        
        if not data or 'complaint_text' not in data:
            return jsonify({
                'error': 'Missing complaint_text in request',
                'status': 'error'
            }), 400
        
        complaint_text = data['complaint_text']
        
        # Validate input
        is_valid, error_msg = validate_complaint_text(complaint_text)
        if not is_valid:
            return jsonify({
                'error': error_msg,
                'status': 'error'
            }), 400
        
        # Make prediction
        logger.info(f"Making prediction for complaint: {complaint_text[:100]}...")
        prediction = model_trainer.predict(complaint_text)
        
        # Prepare response
        response = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'complaint_text': complaint_text,
            'predictions': {
                'category': prediction['category'],
                'priority': prediction['priority'],
                'department': prediction['department'],
                'category_confidence': round(prediction['category_confidence'], 4),
                'priority_confidence': round(prediction['priority_confidence'], 4),
                'category_icon': get_category_icon(prediction['category']),
                'priority_color': get_priority_color(prediction['priority'])
            }
        }
        
        logger.info(f"Prediction completed: {prediction['category']} - {prediction['priority']}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'error': 'An error occurred during prediction. Please try again.',
            'status': 'error',
            'details': str(e)
        }), 500

@app.route('/predict_form', methods=['POST'])
def predict_form():
    """Handle form submission from web interface"""
    try:
        # Check if models are loaded
        if not models_loaded:
            return render_template('index.html', 
                                 error='Models not loaded. Please train the models first.')
        
        # Get form data
        complaint_text = request.form.get('complaint_text', '').strip()
        
        # Validate input
        is_valid, error_msg = validate_complaint_text(complaint_text)
        if not is_valid:
            return render_template('index.html', error=error_msg, 
                                 complaint_text=complaint_text)
        
        # Make prediction
        logger.info(f"Making prediction for form submission: {complaint_text[:100]}...")
        prediction = model_trainer.predict(complaint_text)
        
        # Prepare response data
        result = {
            'complaint_text': complaint_text,
            'category': prediction['category'],
            'priority': prediction['priority'],
            'department': prediction['department'],
            'category_confidence': round(prediction['category_confidence'], 4),
            'priority_confidence': round(prediction['priority_confidence'], 4),
            'category_icon': get_category_icon(prediction['category']),
            'priority_color': get_priority_color(prediction['priority']),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"Form prediction completed: {prediction['category']} - {prediction['priority']}")
        
        return render_template('index.html', result=result, success=True)
        
    except Exception as e:
        logger.error(f"Error in form prediction: {str(e)}")
        logger.error(traceback.format_exc())
        
        return render_template('index.html', 
                             error='An error occurred during prediction. Please try again.',
                             complaint_text=request.form.get('complaint_text', ''))

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model_info')
def model_info():
    """Get model information"""
    try:
        if not models_loaded or not model_trainer:
            return jsonify({
                'error': 'Models not loaded',
                'status': 'error'
            }), 500
        
        info = {
            'status': 'success',
            'model_info': {
                'categories': model_trainer.preprocessor.category_encoder.classes_.tolist(),
                'priorities': model_trainer.preprocessor.priority_encoder.classes_.tolist(),
                'feature_count': len(model_trainer.preprocessor.get_feature_names()),
                'department_mapping': model_trainer.department_mapping
            }
        }
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'error': 'Error retrieving model information',
            'status': 'error'
        }), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('index.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return render_template('index.html', 
                         error='An internal server error occurred. Please try again later.'), 500

@app.errorhandler(400)
def bad_request(error):
    """Handle 400 errors"""
    return render_template('index.html', 
                         error='Bad request. Please check your input.'), 400

def create_app():
    """Create and configure the Flask application"""
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Load models
    load_models()
    
    return app

# Application factory
if __name__ == '__main__':
    app = create_app()
    
    # Run the application
    logger.info("Starting Flask application...")
    logger.info("Access the web interface at: http://127.0.0.1:5000")
    logger.info("API endpoint available at: http://127.0.0.1:5000/predict")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
