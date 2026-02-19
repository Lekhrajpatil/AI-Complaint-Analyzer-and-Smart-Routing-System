"""
Prepare model files for Vercel deployment
Extracts and saves individual model components for serverless deployment
"""

import pickle
import os
from preprocess import ComplaintPreprocessor
from train_model import ComplaintModelTrainer

def prepare_models():
    """Extract and save model components individually"""
    try:
        print("Preparing model files for deployment...")
        
        # Initialize trainer and load existing models
        trainer = ComplaintModelTrainer()
        
        # Load the complete preprocessor
        trainer.preprocessor.load_preprocessor('models/preprocessor.pkl')
        
        # Save individual components to api directory
        api_dir = 'api'
        os.makedirs(api_dir, exist_ok=True)
        
        # Save vectorizer
        with open(os.path.join(api_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(trainer.preprocessor.tfidf_vectorizer, f)
        print("✅ Vectorizer saved to api/vectorizer.pkl")
        
        # Save category model
        with open(os.path.join(api_dir, 'category_model.pkl'), 'wb') as f:
            pickle.dump(trainer.category_model, f)
        print("✅ Category model saved to api/category_model.pkl")
        
        # Save priority model
        with open(os.path.join(api_dir, 'priority_model.pkl'), 'wb') as f:
            pickle.dump(trainer.priority_model, f)
        print("✅ Priority model saved to api/priority_model.pkl")
        
        # Save category encoder
        with open(os.path.join(api_dir, 'category_encoder.pkl'), 'wb') as f:
            pickle.dump(trainer.preprocessor.category_encoder, f)
        print("✅ Category encoder saved to api/category_encoder.pkl")
        
        # Save priority encoder
        with open(os.path.join(api_dir, 'priority_encoder.pkl'), 'wb') as f:
            pickle.dump(trainer.preprocessor.priority_encoder, f)
        print("✅ Priority encoder saved to api/priority_encoder.pkl")
        
        print("\n🎉 Model preparation completed!")
        print("Files ready for Vercel deployment:")
        print("- api/vectorizer.pkl")
        print("- api/category_model.pkl") 
        print("- api/priority_model.pkl")
        print("- api/category_encoder.pkl")
        print("- api/priority_encoder.pkl")
        
    except Exception as e:
        print(f"❌ Error preparing models: {e}")

if __name__ == "__main__":
    prepare_models()
