"""
Synthetic Complaint Dataset Generator
Generates realistic complaint data for training the ML model
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComplaintDatasetGenerator:
    """Generate synthetic complaint dataset with realistic patterns"""
    
    def __init__(self, num_samples=8000):
        self.num_samples = num_samples
        self.categories = ['Billing', 'Technical Issue', 'Service', 'Product', 'Delivery', 'Others']
        self.priorities = ['High', 'Medium', 'Low']
        self.departments = {
            'Billing': 'Finance Department',
            'Technical Issue': 'Technical Support',
            'Service': 'Customer Service',
            'Product': 'Product Team',
            'Delivery': 'Logistics Department',
            'Others': 'General Support'
        }
        
        # Templates for realistic complaint generation with more distinct patterns
        self.complaint_templates = {
            'Billing': [
                "I was charged {amount} for {service} subscription that I cancelled last month",
                "My credit card statement shows unauthorized charge of {amount} from your company",
                "The invoice number {invoice} has incorrect amount {amount} for services not rendered",
                "I'm being double billed for my {service} subscription - charged twice this month",
                "Your billing system charged me {amount} after I downgraded to basic plan",
                "The auto-payment failed but you still charged me late fee of {amount}",
                "My account balance shows {amount} due but I paid on {date}",
                "I need a refund for the wrongful charge of {amount} on {date}",
                "The pricing for {service} is wrong - should be {amount} not {amount}",
                "Your billing department sent me to collections for {amount} I don't owe",
                "I was charged {amount} for premium features I never activated",
                "The subscription renewal charged me {amount} without notification",
                "My bank disputed the charge of {amount} but you haven't responded",
                "The invoice shows {amount} for {service} but I only ordered basic plan",
                "I found fraudulent charges totaling {amount} on my account"
            ],
            'Technical Issue': [
                "The mobile app crashes every time I try to access the {feature} section",
                "I cannot login to my account - getting authentication failed error",
                "The website is down and showing 500 internal server error since {time}",
                "When I upload files to {service}, the system times out after 2 minutes",
                "The password reset link is expired and not working",
                "Your API endpoint for {service} is returning connection timeout errors",
                "The search function on your website is not returning any results",
                "I'm unable to view my reports due to database connection error",
                "The dashboard loading time is extremely slow - takes over 5 minutes",
                "Your system logged me out automatically every 10 minutes",
                "The file upload feature is broken - cannot attach documents",
                "I'm getting access denied error when trying to change my settings",
                "The notification system is not sending email alerts",
                "Your mobile app keeps freezing on the payment screen",
                "The integration with {platform} is not syncing data properly"
            ],
            'Service': [
                "I waited 45 minutes on hold only to be disconnected by customer service",
                "The support representative was rude and refused to help with my {problem}",
                "I was transferred 5 times between departments without resolution",
                "The customer service agent gave me incorrect information about {policy}",
                "No one has responded to my email about {problem} sent {time} ago",
                "The support staff lacks basic knowledge about your {service}",
                "I was promised a callback within 24 hours but never received one",
                "The service quality has declined - wait times increased to {time}",
                "Your representative hung up on me when I asked to speak to a supervisor",
                "Very unprofessional behavior from the customer service team",
                "The support agent couldn't understand my complaint about {problem}",
                "Poor response time - took {time} to get initial reply",
                "No follow-up on my urgent request regarding {problem}",
                "The service team keeps giving me different answers each time I call",
                "Customer service hours are limited and don't work for my schedule"
            ],
            'Product': [
                "The {product} stopped working after just 2 weeks of light use",
                "I received a damaged {product} with cracked screen and missing charger",
                "The {product} I ordered doesn't match the specifications advertised",
                "My new {product} has manufacturing defects - overheating after 10 minutes",
                "The {product} battery drains completely in less than 2 hours",
                "Package arrived with wrong {product} - ordered {product} but received {product}",
                "The {product} firmware is outdated causing constant crashes",
                "My {product} warranty claim was denied without proper explanation",
                "The {product} performance is much worse than your advertised specs",
                "Received {product} with scratches and dents - looks like refurbished",
                "The {product} is not compatible with my {device} as promised",
                "Missing essential accessories in the {product} package",
                "The {product} user manual is in wrong language and unclear",
                "Product quality control is poor - had to return 3 times already",
                "The {product} makes strange noises and doesn't function properly"
            ],
            'Delivery': [
                "My order was supposed to arrive yesterday but tracking shows no movement",
                "Package was delivered to wrong address despite correct shipping information",
                "The express delivery I paid extra for took 5 days instead of promised 2",
                "My package arrived with water damage and items completely ruined",
                "Tracking number shows delivered but I never received anything",
                "The courier left my package outside in the rain without protection",
                "Delivery driver refused to wait for signature and just left the package",
                "My order has been stuck in transit for over {time} with no updates",
                "Package was opened and items were stolen during delivery",
                "Wrong items delivered - I ordered {product} but got {product} instead",
                "The shipping address was correct but delivered to neighbor's house",
                "Express shipping charge applied but standard delivery timeline used",
                "Package was lost by the courier company - no resolution provided",
                "Delivery attempted but no notification left - had to track down myself",
                "The tracking system hasn't updated for {time} - package seems lost"
            ],
            'Others': [
                "I need clarification about your privacy policy regarding data sharing",
                "There is confusion about the return process for international orders",
                "I would like to suggest improvement for your mobile app user interface",
                "General inquiry about your pricing plans and available discounts",
                "Need assistance with account deletion and data export procedures",
                "Question regarding your warranty policy terms and conditions",
                "Concern about recent changes to your terms of service",
                "I have feedback about improving your website navigation",
                "General concern about your company's environmental impact",
                "Need help understanding the cancellation process for {service}",
                "There is lack of transparency about your data security practices",
                "Request for information about corporate social responsibility initiatives",
                "General feedback for improving overall customer experience",
                "Concern regarding your company's response to recent issues",
                "I need help with understanding your loyalty program benefits"
            ]
        }
        
        # Variable replacements for templates
        self.variables = {
            'amount': ['$19.99', '$29.99', '$49.99', '$99.99', '$149.99', '$199.99', '$299.99', '$499.99'],
            'service': ['premium subscription', 'basic plan', 'enterprise package', 'trial service', 'add-on feature'],
            'error': ['404 error', 'timeout error', 'authentication failed', 'server error', 'database error', '500 internal server error', 'connection timeout', 'access denied'],
            'action': ['upload files', 'access dashboard', 'view reports', 'submit form', 'process payment', 'login to account', 'reset password'],
            'feature': ['search function', 'file upload', 'user profile', 'settings menu', 'report generator', 'payment gateway', 'notification system'],
            'time': ['1 day', '2 days', '1 week', '2 weeks', '1 month', '3 months', '30 minutes', '2 hours'],
            'problem': ['billing issue', 'account access', 'service cancellation', 'refund request', 'technical support', 'account verification'],
            'number': ['3', '4', '5', '6', '7'],
            'product': ['laptop', 'smartphone', 'tablet', 'headphones', 'smartwatch', 'camera', 'printer', 'monitor'],
            'issue': ['screen cracks', 'battery drain', 'software glitches', 'hardware failure', 'overheating', 'connection problems'],
            'component': ['screen', 'battery', 'charger', 'cable', 'manual', 'remote control', 'power adapter'],
            'device': ['Windows PC', 'MacBook', 'iPhone', 'Android phone', 'tablet', 'smart TV'],
            'item': ['order #12345', 'package', 'delivery', 'shipment'],
            'policy': ['refund policy', 'privacy policy', 'terms of service', 'warranty policy', 'return policy'],
            'procedure': ['return process', 'cancellation process', 'upgrade process', 'account verification'],
            'request': ['account deletion', 'data export', 'service transfer', 'password reset'],
            'topic': ['account security', 'data privacy', 'service features', 'pricing plans', 'user interface'],
            'company': ['your company', 'the service', 'the platform'],
            'issue': ['communication', 'transparency', 'reliability'],
            'invoice': ['INV-2024-001', 'INV-2024-002', 'INV-2024-003', 'INV-2024-004'],
            'platform': ['mobile app', 'web platform', 'API service', 'third-party integration'],
            'date': ['January 15th', 'February 3rd', 'March 10th', 'April 22nd', 'May 8th']
        }
    
    def generate_complaint_text(self, category):
        """Generate realistic complaint text based on category"""
        templates = self.complaint_templates[category]
        template = random.choice(templates)
        
        # Replace placeholders with realistic values
        for var, values in self.variables.items():
            if f'{{{var}}}' in template:
                template = template.replace(f'{{{var}}}', random.choice(values))
        
        return template
    
    def determine_priority(self, category, text_length):
        """Determine priority based on category and text characteristics"""
        # Base priority probabilities by category
        priority_probs = {
            'Billing': {'High': 0.3, 'Medium': 0.5, 'Low': 0.2},
            'Technical Issue': {'High': 0.4, 'Medium': 0.4, 'Low': 0.2},
            'Service': {'High': 0.2, 'Medium': 0.6, 'Low': 0.2},
            'Product': {'High': 0.3, 'Medium': 0.5, 'Low': 0.2},
            'Delivery': {'High': 0.5, 'Medium': 0.4, 'Low': 0.1},
            'Others': {'High': 0.1, 'Medium': 0.3, 'Low': 0.6}
        }
        
        # Adjust probability based on text length (longer complaints might be more serious)
        if text_length > 200:
            priority_probs[category]['High'] += 0.1
            priority_probs[category]['Low'] -= 0.1
        
        probs = list(priority_probs[category].values())
        return np.random.choice(self.priorities, p=probs)
    
    def generate_dataset(self):
        """Generate complete synthetic dataset"""
        logger.info(f"Generating {self.num_samples} complaint records...")
        
        data = []
        
        for i in range(self.num_samples):
            # Generate complaint details
            category = random.choice(self.categories)
            complaint_text = self.generate_complaint_text(category)
            priority = self.determine_priority(category, len(complaint_text))
            department = self.departments[category]
            
            # Generate metadata
            complaint_id = f"COMP{i+1:06d}"
            date = datetime.now() - timedelta(days=random.randint(0, 365))
            customer_id = f"CUST{random.randint(1000, 9999)}"
            
            data.append({
                'complaint_id': complaint_id,
                'customer_id': customer_id,
                'date': date.strftime('%Y-%m-%d'),
                'complaint_text': complaint_text,
                'category': category,
                'priority': priority,
                'department': department,
                'status': random.choice(['Open', 'In Progress', 'Resolved', 'Closed']),
                'resolution_time': random.randint(1, 72)  # hours
            })
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Generated {i + 1} records...")
        
        df = pd.DataFrame(data)
        logger.info(f"Dataset generation completed. Shape: {df.shape}")
        
        return df
    
    def save_dataset(self, df, filename='complaints_dataset.csv'):
        """Save dataset to CSV file"""
        filepath = f'data/{filename}'
        df.to_csv(filepath, index=False)
        logger.info(f"Dataset saved to {filepath}")
        
        # Display dataset statistics
        logger.info("\nDataset Statistics:")
        logger.info(f"Total records: {len(df)}")
        logger.info("\nCategory distribution:")
        logger.info(df['category'].value_counts())
        logger.info("\nPriority distribution:")
        logger.info(df['priority'].value_counts())
        
        return filepath

def main():
    """Main function to generate and save dataset"""
    try:
        generator = ComplaintDatasetGenerator(num_samples=8000)
        df = generator.generate_dataset()
        filepath = generator.save_dataset(df)
        
        logger.info(f"\nDataset successfully generated and saved to: {filepath}")
        logger.info("Ready for model training!")
        
    except Exception as e:
        logger.error(f"Error generating dataset: {str(e)}")
        raise

if __name__ == "__main__":
    main()
