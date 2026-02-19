"""
Model Training Module for Complaint Analysis
Trains and evaluates ML models for category and priority prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
from typing import Dict, Tuple, Any
import time
from preprocess import ComplaintPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComplaintModelTrainer:
    """Handles model training and evaluation for complaint analysis"""
    
    def __init__(self):
        """Initialize the model trainer"""
        self.preprocessor = ComplaintPreprocessor()
        self.category_model = None
        self.priority_model = None
        self.category_model_name = None
        self.priority_model_name = None
        self.training_history = {}
        
        # Department mapping
        self.department_mapping = {
            'Billing': 'Finance Department',
            'Technical Issue': 'Technical Support',
            'Service': 'Customer Service',
            'Product': 'Product Team',
            'Delivery': 'Logistics Department',
            'Others': 'General Support'
        }
        
        logger.info("Model trainer initialized")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load dataset from CSV file
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Loaded DataFrame
        """
        try:
            logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            
            # Basic data validation
            required_columns = ['complaint_text', 'category', 'priority']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Remove rows with missing values
            df = df.dropna(subset=required_columns)
            logger.info(f"After removing missing values: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def train_category_model(self, X_train, y_train, model_type='random_forest'):
        """
        Train category classification model
        
        Args:
            X_train: Training features
            y_train: Training labels for category
            model_type: Type of model to train ('random_forest' or 'logistic_regression')
        """
        try:
            logger.info(f"Training {model_type} model for category classification...")
            
            if model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2
                )
            elif model_type == 'logistic_regression':
                model = LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    C=1.0
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train the model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            self.category_model = model
            self.category_model_name = model_type
            
            logger.info(f"Category model trained in {training_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error training category model: {str(e)}")
            raise
    
    def train_priority_model(self, X_train, y_train, model_type='random_forest'):
        """
        Train priority classification model
        
        Args:
            X_train: Training features
            y_train: Training labels for priority
            model_type: Type of model to train ('random_forest' or 'logistic_regression')
        """
        try:
            logger.info(f"Training {model_type} model for priority classification...")
            
            if model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=8,
                    min_samples_split=5,
                    min_samples_leaf=2
                )
            elif model_type == 'logistic_regression':
                model = LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    C=1.0
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train the model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            self.priority_model = model
            self.priority_model_name = model_type
            
            logger.info(f"Priority model trained in {training_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error training priority model: {str(e)}")
            raise
    
    def evaluate_model(self, model, X_test, y_test, model_name: str, label_names: list) -> Dict[str, Any]:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model for logging
            label_names: List of label names
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            logger.info(f"Evaluating {model_name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
            
            evaluation_results = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm,
                'cv_mean_score': cv_scores.mean(),
                'cv_std_score': cv_scores.std(),
                'predictions': y_pred
            }
            
            logger.info(f"{model_name} Accuracy: {accuracy:.4f}")
            logger.info(f"{model_name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            raise
    
    def plot_confusion_matrix(self, cm, label_names: list, title: str, save_path: str = None):
        """
        Plot and save confusion matrix
        
        Args:
            cm: Confusion matrix
            label_names: List of label names
            title: Plot title
            save_path: Path to save the plot
        """
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=label_names, yticklabels=label_names)
            plt.title(title)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix saved to {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
    
    def train_and_evaluate(self, filepath: str, test_size: float = 0.2, 
                          category_model_type: str = 'random_forest',
                          priority_model_type: str = 'random_forest'):
        """
        Complete training and evaluation pipeline
        
        Args:
            filepath: Path to the dataset
            test_size: Proportion of data for testing
            category_model_type: Model type for category classification
            priority_model_type: Model type for priority classification
        """
        try:
            logger.info("Starting complete training and evaluation pipeline...")
            
            # Load data
            df = self.load_data(filepath)
            
            # Preprocess data
            X, y_category, y_priority = self.preprocessor.fit_transform(df)
            
            # Split data
            X_train, X_test, y_cat_train, y_cat_test = train_test_split(
                X, y_category, test_size=test_size, random_state=42, stratify=y_category
            )
            X_train, X_test, y_pri_train, y_pri_test = train_test_split(
                X, y_priority, test_size=test_size, random_state=42, stratify=y_priority
            )
            
            logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
            
            # Train models
            self.train_category_model(X_train, y_cat_train, category_model_type)
            self.train_priority_model(X_train, y_pri_train, priority_model_type)
            
            # Evaluate models
            category_eval = self.evaluate_model(
                self.category_model, X_test, y_cat_test, 
                "Category Model", self.preprocessor.category_encoder.classes_
            )
            
            priority_eval = self.evaluate_model(
                self.priority_model, X_test, y_pri_test,
                "Priority Model", self.preprocessor.priority_encoder.classes_
            )
            
            # Store training history
            self.training_history = {
                'category_evaluation': category_eval,
                'priority_evaluation': priority_eval,
                'category_model_type': category_model_type,
                'priority_model_type': priority_model_type,
                'training_samples': X_train.shape[0],
                'test_samples': X_test.shape[0]
            }
            
            # Plot confusion matrices
            self.plot_confusion_matrix(
                category_eval['confusion_matrix'],
                self.preprocessor.category_encoder.classes_,
                'Category Classification Confusion Matrix',
                'static/category_confusion_matrix.png'
            )
            
            self.plot_confusion_matrix(
                priority_eval['confusion_matrix'],
                self.preprocessor.priority_encoder.classes_,
                'Priority Classification Confusion Matrix',
                'static/priority_confusion_matrix.png'
            )
            
            # Print detailed results
            self.print_evaluation_results()
            
            logger.info("Training and evaluation completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in training and evaluation: {str(e)}")
            raise
    
    def print_evaluation_results(self):
        """Print detailed evaluation results"""
        try:
            if not self.training_history:
                logger.warning("No training history available")
                return
            
            cat_eval = self.training_history['category_evaluation']
            pri_eval = self.training_history['priority_evaluation']
            
            print("\n" + "="*60)
            print("MODEL EVALUATION RESULTS")
            print("="*60)
            
            print(f"\nCategory Model: {self.training_history['category_model_type']}")
            print(f"Accuracy: {cat_eval['accuracy']:.4f}")
            print(f"Cross-Validation Score: {cat_eval['cv_mean_score']:.4f} (+/- {cat_eval['cv_std_score']*2:.4f})")
            
            print(f"\nPriority Model: {self.training_history['priority_model_type']}")
            print(f"Accuracy: {pri_eval['accuracy']:.4f}")
            print(f"Cross-Validation Score: {pri_eval['cv_mean_score']:.4f} (+/- {pri_eval['cv_std_score']*2:.4f})")
            
            print("\n" + "="*60)
            print("DETAILED CLASSIFICATION REPORTS")
            print("="*60)
            
            print("\nCategory Classification Report:")
            for label, metrics in cat_eval['classification_report'].items():
                if isinstance(metrics, dict):
                    print(f"{label}:")
                    print(f"  Precision: {metrics['precision']:.4f}")
                    print(f"  Recall: {metrics['recall']:.4f}")
                    print(f"  F1-Score: {metrics['f1-score']:.4f}")
            
            print("\nPriority Classification Report:")
            for label, metrics in pri_eval['classification_report'].items():
                if isinstance(metrics, dict):
                    print(f"{label}:")
                    print(f"  Precision: {metrics['precision']:.4f}")
                    print(f"  Recall: {metrics['recall']:.4f}")
                    print(f"  F1-Score: {metrics['f1-score']:.4f}")
            
            print("\n" + "="*60)
            
        except Exception as e:
            logger.error(f"Error printing evaluation results: {str(e)}")
    
    def predict(self, complaint_text: str) -> Dict[str, str]:
        """
        Make predictions for a single complaint
        
        Args:
            complaint_text: Raw complaint text
            
        Returns:
            Dictionary with predictions
        """
        try:
            if not self.category_model or not self.priority_model:
                raise ValueError("Models not trained yet")
            
            # Preprocess text
            processed_features = self.preprocessor.transform([complaint_text])
            
            # Make predictions
            category_pred = self.category_model.predict(processed_features)[0]
            priority_pred = self.priority_model.predict(processed_features)[0]
            
            # Convert back to original labels
            category_label = self.preprocessor.inverse_transform_category([category_pred])[0]
            priority_label = self.preprocessor.inverse_transform_priority([priority_pred])[0]
            
            # Get department
            department = self.department_mapping.get(category_label, 'General Support')
            
            # Get prediction probabilities
            category_proba = self.category_model.predict_proba(processed_features)[0]
            priority_proba = self.priority_model.predict_proba(processed_features)[0]
            
            category_confidence = max(category_proba)
            priority_confidence = max(priority_proba)
            
            return {
                'category': category_label,
                'priority': priority_label,
                'department': department,
                'category_confidence': float(category_confidence),
                'priority_confidence': float(priority_confidence)
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def save_models(self, preprocessor_path: str, category_model_path: str, priority_model_path: str):
        """
        Save trained models and preprocessor
        
        Args:
            preprocessor_path: Path to save preprocessor
            category_model_path: Path to save category model
            priority_model_path: Path to save priority model
        """
        try:
            # Save preprocessor
            self.preprocessor.save_preprocessor(preprocessor_path)
            
            # Save models
            with open(category_model_path, 'wb') as f:
                pickle.dump(self.category_model, f)
            
            with open(priority_model_path, 'wb') as f:
                pickle.dump(self.priority_model, f)
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
    
    def load_models(self, preprocessor_path: str, category_model_path: str, priority_model_path: str):
        """
        Load trained models and preprocessor
        
        Args:
            preprocessor_path: Path to load preprocessor
            category_model_path: Path to load category model
            priority_model_path: Path to load priority model
        """
        try:
            # Load preprocessor
            self.preprocessor.load_preprocessor(preprocessor_path)
            
            # Load models
            with open(category_model_path, 'rb') as f:
                self.category_model = pickle.load(f)
            
            with open(priority_model_path, 'rb') as f:
                self.priority_model = pickle.load(f)
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

def main():
    """Main function to train and evaluate models"""
    try:
        # Initialize trainer
        trainer = ComplaintModelTrainer()
        
        # Train and evaluate models
        trainer.train_and_evaluate(
            filepath='data/complaints_dataset.csv',
            category_model_type='random_forest',
            priority_model_type='random_forest'
        )
        
        # Save models
        trainer.save_models(
            'models/preprocessor.pkl',
            'models/category_model.pkl',
            'models/priority_model.pkl'
        )
        
        # Test prediction
        test_complaint = "I was charged $99.99 for a service I didn't subscribe to and I can't login to my account"
        prediction = trainer.predict(test_complaint)
        
        print(f"\nTest Prediction:")
        print(f"Complaint: {test_complaint}")
        print(f"Category: {prediction['category']} (Confidence: {prediction['category_confidence']:.4f})")
        print(f"Priority: {prediction['priority']} (Confidence: {prediction['priority_confidence']:.4f})")
        print(f"Department: {prediction['department']}")
        
    except Exception as e:
        logger.error(f"Error in main training process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
