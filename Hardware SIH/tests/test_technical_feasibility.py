import unittest
import numpy as np
import pandas as pd
from technical_feasibility import TechnicalFeasibilityEvaluator, create_sample_training_data

class TestTechnicalFeasibilityEvaluator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.evaluator = TechnicalFeasibilityEvaluator()
        self.sample_proposal = {
            'Title': 'AI for Mine Safety Monitoring',
            'Abstract': 'Using artificial intelligence to monitor and predict safety hazards in coal mines to prevent accidents with a detailed technical approach.',
            'Funding_Requested': 80000
        }
    
    def test_initialization(self):
        """Test that the evaluator initializes correctly."""
        self.assertIsNotNone(self.evaluator)
        self.assertIsNone(self.evaluator.xgb_model)
        self.assertIsNone(self.evaluator.lgb_model)
        self.assertIsNotNone(self.evaluator.tfidf_vectorizer)
    
    def test_extract_numerical_features(self):
        """Test numerical feature extraction."""
        # Create sample data
        df = pd.DataFrame([self.sample_proposal])
        
        # Extract features
        features = self.evaluator.extract_numerical_features(df)
        
        # Check that we get the expected columns
        expected_columns = [
            'team_experience_score', 
            'budget_realism_score', 
            'timeline_realism_score',
            'methodology_completeness_score'
        ]
        
        self.assertEqual(list(features.columns), expected_columns)
        self.assertEqual(len(features), 1)
        
        # Check that all values are between 0 and 1
        for col in features.columns:
            self.assertTrue(0 <= features.iloc[0][col] <= 1)
    
    def test_calculate_team_experience(self):
        """Test team experience calculation."""
        score = self.evaluator._calculate_team_experience(
            "Experienced Research Team", 
            "Our team has extensive experience in machine learning and data science with multiple PhDs."
        )
        self.assertIsInstance(score, float)
        self.assertTrue(0 <= score <= 1)
    
    def test_calculate_methodology_completeness(self):
        """Test methodology completeness calculation."""
        score = self.evaluator._calculate_methodology_completeness(
            "Our approach includes detailed methodology with experiments, data collection, and validation procedures."
        )
        self.assertIsInstance(score, float)
        self.assertTrue(0 <= score <= 1)
    
    def test_prepare_text_features(self):
        """Test text feature preparation."""
        texts = [
            "Machine learning for image recognition",
            "Cloud computing infrastructure",
            "Data mining techniques"
        ]
        
        features = self.evaluator.prepare_text_features(texts)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], 3)  # 3 texts
        self.assertEqual(features.shape[1], 1000)  # 1000 features (max_features)
    
    def test_evaluate_proposal(self):
        """Test proposal evaluation."""
        result = self.evaluator.evaluate_proposal(self.sample_proposal)
        
        # Check that we get expected keys
        expected_keys = [
            'feasibility_score',
            'team_experience_score',
            'budget_realism_score',
            'timeline_realism_score',
            'methodology_completeness_score',
            'bert_analysis_score'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result)
            self.assertIsInstance(result[key], float)
            self.assertTrue(0 <= result[key] <= 1)
    
    def test_create_sample_training_data(self):
        """Test sample training data creation."""
        df, labels = create_sample_training_data()
        
        # Check data structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(df), len(labels))
        self.assertGreater(len(df), 0)
        
        # Check required columns
        required_columns = ['Proposal_ID', 'Title', 'Abstract', 'Funding_Requested']
        for col in required_columns:
            self.assertIn(col, df.columns)

if __name__ == '__main__':
    unittest.main()