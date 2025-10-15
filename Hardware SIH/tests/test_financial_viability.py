import unittest
import pandas as pd
import numpy as np
from financial_viability import FinancialViabilityEvaluator, create_sample_budget_data

class TestFinancialViabilityEvaluator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.evaluator = FinancialViabilityEvaluator()
        self.sample_proposal = {
            'proposal_id': 'TEST001',
            'title': 'Test Proposal',
            'funding_requested': 100000,
            'budget_breakdown': {
                'personnel': 60000,
                'equipment': 30000,
                'travel': 5000,
                'overhead': 5000
            },
            'duration_months': 12,
            'personnel_count': 3,
            'technical_complexity': 5
        }
    
    def test_initialization(self):
        """Test that the evaluator initializes correctly."""
        self.assertIsNotNone(self.evaluator)
        self.assertIsNone(self.evaluator.isolation_forest)
        self.assertFalse(self.evaluator.trained)
    
    def test_validate_budget_against_guidelines_valid(self):
        """Test budget validation with valid budget."""
        budget_data = {
            'personnel': 60000,  # 60% of total
            'equipment': 30000,  # 30% of total
            'travel': 5000,      # 5% of total
            'overhead': 5000     # 5% of total
        }
        
        result = self.evaluator.validate_budget_against_guidelines(budget_data)
        
        self.assertTrue(result['valid'])
        self.assertEqual(result['total_funding'], 100000)
        self.assertEqual(len(result['violations']), 0)
    
    def test_validate_budget_against_guidelines_invalid(self):
        """Test budget validation with invalid budget."""
        budget_data = {
            'personnel': 20000,   # 20% of total (below minimum 50%)
            'equipment': 50000,   # 50% of total (above maximum 40%)
            'overhead': 30000     # 30% of total (above maximum 20%)
        }
        
        result = self.evaluator.validate_budget_against_guidelines(budget_data)
        
        self.assertFalse(result['valid'])
        self.assertEqual(result['total_funding'], 100000)
        self.assertGreater(len(result['violations']), 0)
        
        # Check for specific violations
        violation_rules = [v['rule'] for v in result['violations']]
        self.assertIn('equipment_percentage_cap', violation_rules)
        self.assertIn('personnel_percentage_floor', violation_rules)
        self.assertIn('overhead_percentage_cap', violation_rules)
    
    def test_calculate_expected_budget(self):
        """Test expected budget calculation."""
        features = {
            'duration_months': 12,
            'personnel_count': 3,
            'technical_complexity': 5
        }
        
        expected_budget = self.evaluator.calculate_expected_budget(features)
        self.assertIsInstance(expected_budget, float)
        self.assertGreater(expected_budget, 0)
    
    def test_calculate_budget_zscore(self):
        """Test budget z-score calculation."""
        z_score = self.evaluator.calculate_budget_zscore(100000, 80000)
        self.assertIsInstance(z_score, float)
        
        # Test with same values (should be 0)
        z_score_zero = self.evaluator.calculate_budget_zscore(80000, 80000)
        self.assertEqual(z_score_zero, 0)
    
    def test_evaluate_proposal(self):
        """Test proposal evaluation."""
        result = self.evaluator.evaluate_proposal(self.sample_proposal)
        
        # Check that we get expected keys
        expected_keys = [
            'financial_score',
            'guideline_validation',
            'expected_budget',
            'actual_budget',
            'budget_zscore',
            'anomaly_detection',
            'recommendations'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check value types
        self.assertIsInstance(result['financial_score'], float)
        self.assertTrue(0 <= result['financial_score'] <= 1)
        self.assertIsInstance(result['actual_budget'], (int, float))
        self.assertIsInstance(result['expected_budget'], (int, float))
        self.assertIsInstance(result['budget_zscore'], float)
    
    def test_train_anomaly_detector(self):
        """Test anomaly detector training."""
        # Create sample data
        sample_data = create_sample_budget_data()
        historical_df = pd.DataFrame(sample_data[:-1])
        
        # Train detector
        self.evaluator.train_anomaly_detector(historical_df)
        
        # Check that it's trained
        self.assertTrue(self.evaluator.trained)
        self.assertIsNotNone(self.evaluator.isolation_forest)
    
    def test_detect_budget_anomalies(self):
        """Test anomaly detection."""
        # Create sample data
        sample_data = create_sample_budget_data()
        historical_df = pd.DataFrame(sample_data[:-1])
        
        # Train detector
        self.evaluator.train_anomaly_detector(historical_df)
        
        # Test detection
        anomaly_result = self.evaluator.detect_budget_anomalies(self.sample_proposal)
        
        # Check results
        self.assertIsInstance(anomaly_result, dict)
        self.assertIn('is_anomaly', anomaly_result)
        self.assertIn('anomaly_score', anomaly_result)
        self.assertIn('confidence', anomaly_result)

if __name__ == '__main__':
    unittest.main()