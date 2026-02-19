"""
Test script for the AI Complaint Analyzer API
"""

import requests
import json

def test_api():
    """Test the complaint analysis API"""
    
    # API endpoint
    url = "http://127.0.0.1:5000/predict"
    
    # Test cases
    test_cases = [
        {
            "name": "Billing Complaint",
            "text": "I was charged $99.99 for a premium subscription that I cancelled last month"
        },
        {
            "name": "Technical Issue",
            "text": "The mobile app crashes every time I try to access the payment gateway"
        },
        {
            "name": "Service Complaint",
            "text": "I waited 45 minutes on hold only to be disconnected by customer service"
        },
        {
            "name": "Product Issue",
            "text": "The laptop I received has a cracked screen and missing charger"
        },
        {
            "name": "Delivery Problem",
            "text": "My package was delivered to wrong address despite correct shipping information"
        }
    ]
    
    print("Testing AI Complaint Analyzer API")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Input: {test_case['text']}")
        
        try:
            # Send request
            response = requests.post(url, json={"complaint_text": test_case['text']})
            
            if response.status_code == 200:
                result = response.json()
                if result['status'] == 'success':
                    predictions = result['predictions']
                    print(f"✅ Category: {predictions['category']} (Confidence: {predictions['category_confidence']:.4f})")
                    print(f"✅ Priority: {predictions['priority']} (Confidence: {predictions['priority_confidence']:.4f})")
                    print(f"✅ Department: {predictions['department']}")
                else:
                    print(f"❌ API Error: {result.get('error', 'Unknown error')}")
            else:
                print(f"❌ HTTP Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection Error: {str(e)}")
        except Exception as e:
            print(f"❌ Unexpected Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("API Testing Complete!")

if __name__ == "__main__":
    test_api()
