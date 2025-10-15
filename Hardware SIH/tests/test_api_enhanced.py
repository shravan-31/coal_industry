import unittest
import json
import os
import pandas as pd
from api_enhanced import app

class TestEnhancedAPI(unittest.TestCase):
    
    def setUp(self):
        """Set up test client and sample data before each test."""
        self.app = app.test_client()
        self.app.testing = True
        
        # Create sample data files if they don't exist
        self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data files for testing."""
        # Sample past proposals
        past_data = {
            'Proposal_ID': ['P001', 'P002'],
            'Title': [
                'Machine Learning for Image Recognition',
                'Cloud Computing Infrastructure'
            ],
            'Abstract': [
                'This project focuses on developing advanced machine learning algorithms for image recognition tasks.',
                'Research on scalable cloud computing infrastructure for enterprise applications.'
            ],
            'Funding_Requested': [50000, 75000]
        }
        
        past_df = pd.DataFrame(past_data)
        past_df.to_csv('sample_past_proposals.csv', index=False)
        
        # Sample new proposals
        new_data = {
            'Proposal_ID': ['N001'],
            'Title': ['AI for Mine Safety Monitoring'],
            'Abstract': ['Using artificial intelligence to monitor and predict safety hazards in coal mines.'],
            'Funding_Requested': [80000]
        }
        
        new_df = pd.DataFrame(new_data)
        new_df.to_csv('sample_new_proposals.csv', index=False)
    
    def tearDown(self):
        """Clean up test files after each test."""
        test_files = [
            'sample_past_proposals.csv',
            'sample_new_proposals.csv',
            'evaluated_proposals.csv'
        ]
        
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('message', data)
    
    def test_get_current_weights(self):
        """Test getting current weights."""
        response = self.app.get('/weights')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('weights', data)
        
        weights = data['weights']
        self.assertIsInstance(weights, dict)
        self.assertIn('novelty', weights)
        self.assertIn('financial', weights)
    
    def test_update_weights_valid(self):
        """Test updating weights with valid data."""
        new_weights = {
            'weights': {
                'novelty': 0.30,
                'financial': 0.20,
                'technical': 0.20,
                'coal_relevance': 0.10,
                'alignment': 0.10,
                'clarity': 0.05,
                'impact': 0.05
            }
        }
        
        response = self.app.post('/weights',
                               data=json.dumps(new_weights),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('weights', data)
    
    def test_update_weights_invalid(self):
        """Test updating weights with invalid data."""
        invalid_weights = {
            'weights': {
                'novelty': 0.50,
                'financial': 0.50,
                'technical': 0.50  # This makes sum > 1.0
            }
        }
        
        response = self.app.post('/weights',
                               data=json.dumps(invalid_weights),
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'error')
    
    def test_evaluate_proposals(self):
        """Test evaluating proposals."""
        evaluation_data = {
            'past_proposals_path': 'sample_past_proposals.csv',
            'new_proposals_path': 'sample_new_proposals.csv'
        }
        
        response = self.app.post('/proposals/evaluate',
                               data=json.dumps(evaluation_data),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('results', data)
        self.assertIsInstance(data['results'], list)
    
    def test_get_proposal_report(self):
        """Test getting proposal report."""
        response = self.app.get('/proposals/N001/report')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('report', data)
        
        report = data['report']
        self.assertEqual(report['proposal_id'], 'N001')
        self.assertIn('technical_feasibility', report)
        self.assertIn('financial_viability', report)
    
    def test_submit_feedback(self):
        """Test submitting feedback."""
        feedback_data = {
            'reviewer_id': 'TEST_REVIEWER',
            'feedback': {
                'comments': 'Excellent proposal',
                'rating': 4.5,
                'accept': True
            }
        }
        
        response = self.app.post('/proposals/N001/feedback',
                               data=json.dumps(feedback_data),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('message', data)
    
    def test_search_similar_proposals(self):
        """Test searching similar proposals."""
        response = self.app.get('/search/similar?proposal_id=N001&k=3')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('similar_proposals', data)
        self.assertIsInstance(data['similar_proposals'], list)
    
    def test_get_monitoring_report(self):
        """Test getting monitoring report."""
        response = self.app.get('/monitoring/report')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('monitoring_report', data)
    
    def test_get_monitoring_alerts(self):
        """Test getting monitoring alerts."""
        response = self.app.get('/monitoring/alerts')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('alerts', data)
        self.assertIn('retraining_needed', data)
    
    def test_explain_evaluation(self):
        """Test getting evaluation explanation."""
        response = self.app.get('/explain/N001')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('explanation', data)
        
        explanation = data['explanation']
        self.assertEqual(explanation['proposal_id'], 'N001')
        self.assertIn('shap_explanation', explanation)
        self.assertIn('lime_explanation', explanation)
    
    def test_generate_sample_data(self):
        """Test generating sample data."""
        # Remove existing sample files
        sample_files = ['sample_past_proposals.csv', 'sample_new_proposals.csv']
        for file in sample_files:
            if os.path.exists(file):
                os.remove(file)
        
        response = self.app.post('/sample-data')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        
        # Check that files were created
        for file in sample_files:
            self.assertTrue(os.path.exists(file))
    
    def test_upload_proposal_no_file(self):
        """Test uploading proposal without file."""
        response = self.app.post('/proposals/upload')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'error')
        self.assertIn('message', data)

if __name__ == '__main__':
    unittest.main()