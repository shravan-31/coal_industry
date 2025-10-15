import unittest
import pandas as pd
import os
from enhanced_evaluator_hil import EnhancedRDPEvaluatorHIL

class TestEnhancedRDPEvaluatorHIL(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.evaluator = EnhancedRDPEvaluatorHIL()
        
        # Create sample data for testing
        self.create_sample_data()
    
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Remove test files
        test_files = [
            'test_past_proposals.csv', 
            'test_new_proposals.csv',
            'evaluator_feedback.json',
            'test_evaluated_proposals.csv'
        ]
        
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
    
    def create_sample_data(self):
        """Create sample data for testing."""
        # Sample past proposals
        past_data = {
            'Proposal_ID': ['P001', 'P002', 'P003'],
            'Title': [
                'Machine Learning for Image Recognition',
                'Cloud Computing Infrastructure',
                'Data Mining Techniques'
            ],
            'Abstract': [
                'This project focuses on developing advanced machine learning algorithms for image recognition tasks.',
                'Research on scalable cloud computing infrastructure for enterprise applications.',
                'Exploring data mining techniques for extracting valuable insights from large datasets.'
            ],
            'Funding_Requested': [50000, 75000, 30000]
        }
        
        past_df = pd.DataFrame(past_data)
        past_df.to_csv('test_past_proposals.csv', index=False)
        
        # Sample new proposals
        new_data = {
            'Proposal_ID': ['N001', 'N002'],
            'Title': [
                'AI for Mine Safety Monitoring',
                'Coal Quality Prediction Using Machine Learning'
            ],
            'Abstract': [
                'Using artificial intelligence to monitor and predict safety hazards in coal mines to prevent accidents.',
                'Developing machine learning models to predict coal quality based on geological and chemical data.'
            ],
            'Funding_Requested': [80000, 60000]
        }
        
        new_df = pd.DataFrame(new_data)
        new_df.to_csv('test_new_proposals.csv', index=False)
    
    def test_initialization(self):
        """Test that the evaluator initializes correctly."""
        self.assertIsNotNone(self.evaluator)
        self.assertEqual(self.evaluator.novelty_weight, 0.20)
        self.assertEqual(self.evaluator.financial_weight, 0.15)
    
    def test_get_weights(self):
        """Test getting weights."""
        weights = self.evaluator.get_weights()
        
        self.assertIsInstance(weights, dict)
        self.assertIn('novelty', weights)
        self.assertIn('financial', weights)
        self.assertIn('technical', weights)
        self.assertEqual(sum(weights.values()), 1.0)
    
    def test_set_weights_valid(self):
        """Test setting valid weights."""
        new_weights = {
            'novelty': 0.30,
            'financial': 0.20,
            'technical': 0.20,
            'coal_relevance': 0.10,
            'alignment': 0.10,
            'clarity': 0.05,
            'impact': 0.05
        }
        
        result = self.evaluator.set_weights(new_weights)
        
        self.assertTrue(result)
        self.assertEqual(self.evaluator.novelty_weight, 0.30)
        self.assertEqual(self.evaluator.financial_weight, 0.20)
    
    def test_set_weights_invalid(self):
        """Test setting invalid weights (don't sum to 1.0)."""
        new_weights = {
            'novelty': 0.50,
            'financial': 0.50,
            'technical': 0.50  # This makes sum > 1.0
        }
        
        result = self.evaluator.set_weights(new_weights)
        
        self.assertFalse(result)
        # Weights should remain unchanged
        self.assertEqual(self.evaluator.novelty_weight, 0.20)
    
    def test_evaluate_proposals(self):
        """Test evaluating proposals."""
        results = self.evaluator.evaluate_proposals(
            'test_past_proposals.csv',
            'test_new_proposals.csv'
        )
        
        self.assertIsInstance(results, pd.DataFrame)
        self.assertGreater(len(results), 0)
        self.assertIn('Overall_Score', results.columns)
        self.assertIn('Recommendation', results.columns)
    
    def test_collect_human_feedback(self):
        """Test collecting human feedback."""
        feedback = {
            "comments": "Excellent proposal with strong methodology",
            "rating": 4.8,
            "accept": True
        }
        
        self.evaluator.collect_human_feedback(
            proposal_id="N001",
            reviewer_id="REV001",
            feedback=feedback
        )
        
        # Check that feedback was stored
        stored_feedback = self.evaluator.get_feedback_for_proposal("N001")
        self.assertEqual(len(stored_feedback), 1)
        self.assertEqual(stored_feedback[0]["reviewer_id"], "REV001")
    
    def test_get_feedback_for_proposal(self):
        """Test getting feedback for a proposal."""
        # Add some feedback
        feedback1 = {"comments": "Good proposal", "rating": 4.0, "accept": True}
        feedback2 = {"comments": "Needs improvement", "rating": 3.0, "accept": False}
        
        self.evaluator.collect_human_feedback("N001", "REV001", feedback1)
        self.evaluator.collect_human_feedback("N001", "REV002", feedback2)
        self.evaluator.collect_human_feedback("N002", "REV003", {"comments": "OK", "rating": 3.5, "accept": True})
        
        # Get feedback for specific proposal
        proposal_feedback = self.evaluator.get_feedback_for_proposal("N001")
        
        self.assertEqual(len(proposal_feedback), 2)
        self.assertEqual(proposal_feedback[0]["proposal_id"], "N001")
    
    def test_save_and_load_feedback(self):
        """Test saving and loading feedback."""
        # Add feedback
        feedback = {"comments": "Test feedback", "rating": 4.5, "accept": True}
        self.evaluator.collect_human_feedback("N001", "REV001", feedback)
        
        # Create new evaluator instance to test loading
        new_evaluator = EnhancedRDPEvaluatorHIL()
        
        # Check that feedback was loaded
        loaded_feedback = new_evaluator.get_feedback_for_proposal("N001")
        self.assertEqual(len(loaded_feedback), 1)
        self.assertEqual(loaded_feedback[0]["reviewer_id"], "REV001")
    
    def test_save_results(self):
        """Test saving evaluation results."""
        # Evaluate proposals
        results = self.evaluator.evaluate_proposals(
            'test_past_proposals.csv',
            'test_new_proposals.csv'
        )
        
        # Save results
        output_path = 'test_evaluated_proposals.csv'
        self.evaluator.save_results(results, output_path)
        
        # Check that file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Check that it can be read as CSV
        saved_results = pd.read_csv(output_path)
        self.assertGreater(len(saved_results), 0)

if __name__ == '__main__':
    unittest.main()