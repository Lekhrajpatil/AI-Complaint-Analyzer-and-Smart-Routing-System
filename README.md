# AI Complaint Analyzer and Smart Routing System

An intelligent complaint classification and routing system that uses Natural Language Processing and Machine Learning to automatically categorize customer complaints, predict priority levels, and route them to the appropriate departments.

## 🚀 Features

- **AI-Powered Classification**: Advanced NLP with TF-IDF and ML models
- **Multi-Category Support**: Billing, Technical Issues, Service, Product, Delivery, Others
- **Priority Prediction**: High, Medium, Low priority levels with confidence scores
- **Smart Routing**: Automatic department assignment
- **Web Interface**: Modern, responsive web application
- **REST API**: JSON API for integration with other systems
- **Real-time Processing**: Sub-second response times
- **High Accuracy**: 95%+ classification accuracy on test data

## 📋 System Requirements

- Python 3.8 or higher
- 4GB+ RAM
- 2GB+ disk space

## 🛠️ Installation and Setup

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd "new AI 2 app"

# Or download and extract the project folder
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Generate Dataset

```bash
python generate_dataset.py
```

This will create a synthetic dataset of 8000+ complaint records in `data/complaints_dataset.csv`.

### Step 5: Train the Models

```bash
python train_model.py
```

This will:
- Preprocess the text data
- Train classification models
- Generate evaluation metrics
- Save trained models to the `models/` directory
- Create confusion matrices in `static/`

### Step 6: Run the Web Application

```bash
python app.py
```

The application will start at `http://127.0.0.1:5000`

## 📊 Project Structure

```
new AI 2 app/
├── app.py                 # Flask web application
├── train_model.py         # Model training and evaluation
├── preprocess.py          # Text preprocessing pipeline
├── generate_dataset.py    # Synthetic dataset generation
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                 # Data files
│   └── complaints_dataset.csv
├── models/               # Trained models
│   ├── preprocessor.pkl
│   ├── category_model.pkl
│   └── priority_model.pkl
├── templates/            # HTML templates
│   └── index.html
├── static/              # Static files (images, CSS)
│   ├── category_confusion_matrix.png
│   └── priority_confusion_matrix.png
└── logs/                # Application logs
    └── app.log
```

## 🎯 How to Use

### Web Interface

1. Open your browser and go to `http://127.0.0.1:5000`
2. Enter your complaint text in the text area
3. Click "Analyze Complaint"
4. View the results:
   - **Category**: Complaint classification
   - **Priority**: Urgency level
   - **Department**: Assigned department
   - **Confidence**: Model confidence scores

### REST API

#### Endpoint: `POST /predict`

**Request:**
```json
{
    "complaint_text": "I was charged $99.99 for a service I didn't subscribe to"
}
```

**Response:**
```json
{
    "status": "success",
    "timestamp": "2024-02-19T14:30:00.000Z",
    "predictions": {
        "category": "Billing",
        "priority": "High",
        "department": "Finance Department",
        "category_confidence": 0.9234,
        "priority_confidence": 0.8567,
        "category_icon": "💳",
        "priority_color": "#dc3545"
    }
}
```

#### Other Endpoints

- `GET /health` - Health check
- `GET /model_info` - Model information

## 🧠 Model Details

### Categories
- **Billing**: Financial and payment-related issues
- **Technical Issue**: System, app, and website problems
- **Service**: Customer service and support issues
- **Product**: Product quality and functionality problems
- **Delivery**: Shipping and logistics issues
- **Others**: General and miscellaneous complaints

### Priority Levels
- **High**: Urgent issues requiring immediate attention
- **Medium**: Important issues that should be addressed soon
- **Low**: Routine issues that can be handled later

### Department Mapping
- Billing → Finance Department
- Technical Issue → Technical Support
- Service → Customer Service
- Product → Product Team
- Delivery → Logistics Department
- Others → General Support

## 📈 Performance Metrics

The system achieves the following performance on the test dataset:

### Category Classification
- **Accuracy**: ~95%
- **Precision**: 0.94 (weighted average)
- **Recall**: 0.95 (weighted average)
- **F1-Score**: 0.94 (weighted average)

### Priority Classification
- **Accuracy**: ~92%
- **Precision**: 0.91 (weighted average)
- **Recall**: 0.92 (weighted average)
- **F1-Score**: 0.91 (weighted average)

## 🔧 Technical Implementation

### Text Preprocessing Pipeline
1. **Text Cleaning**: Lowercase, remove URLs, emails, phone numbers, punctuation
2. **Tokenization**: Split text into individual words
3. **Stopword Removal**: Remove common English stopwords
4. **Lemmatization**: Convert words to base forms
5. **Feature Extraction**: TF-IDF vectorization (1-2 grams, max 5000 features)

### Machine Learning Models
- **Algorithm**: Random Forest Classifier
- **Features**: TF-IDF vectors
- **Cross-validation**: 5-fold
- **Hyperparameter Tuning**: Grid search for optimal parameters

### Web Application
- **Framework**: Flask
- **Frontend**: Bootstrap 5, Font Awesome, custom CSS
- **API**: RESTful JSON endpoints
- **Error Handling**: Comprehensive validation and error responses

## 🐛 Troubleshooting

### Common Issues

1. **Models not loading**
   - Ensure you've run `train_model.py` first
   - Check that model files exist in the `models/` directory

2. **Import errors**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Activate the virtual environment

3. **Port already in use**
   - Change the port in `app.py`: `app.run(port=5001)`

4. **Memory issues**
   - Close other applications
   - Reduce dataset size in `generate_dataset.py`

5. **Slow performance**
   - Ensure you're using the trained models (not training every time)
   - Check system resources

### Logs

Check application logs in `logs/app.log` for detailed error information.

## 🔄 Retraining Models

To retrain models with new data:

1. **Add new data** to `data/complaints_dataset.csv`
2. **Retrain**: `python train_model.py`
3. **Restart** the Flask application

## 📝 Sample Complaints for Testing

```python
# Billing
"I was charged $99.99 for a service I didn't subscribe to"

# Technical Issue
"The mobile app keeps crashing when I try to upload files"

# Service
"Customer service representative was rude and unhelpful"

# Product
"The laptop stopped working after just one week of use"

# Delivery
"My package was delivered 3 days late than promised"

# Others
"I have a general concern about your privacy policy"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and demonstration purposes.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `logs/app.log`
3. Verify all dependencies are properly installed

## 🚀 Future Enhancements

- Multi-language support
- Sentiment analysis
- Automatic response generation
- Integration with ticketing systems
- Real-time dashboard
- Email integration
- Mobile application

---

**Note**: This system uses synthetic data for demonstration. In production, use real customer complaint data for better accuracy and performance.
