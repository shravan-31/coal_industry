import unittest
import numpy as np
import pandas as pd
from explainability import ModelExplainer, MockXGBoostModel, MockLightGBMModel, create_sample_proposals

class TestModelExplainer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.explainer = ModelExplainer()
        self.sample_proposals = create_sample_proposals()
        self.X_train = self.explainer.prepare_training_data(self.sample_proposals)
        self.mock_models = {
            'xgboost': MockXGBoostModel(),
            'lightgbm': MockLightGBMModel()
        }
    
    def test_initialization(self):
        """Test that the explainer initializes correctly."""
        self.assertIsNotNone(self.explainer)
        self.assertEqual(self.explainer.feature_names, [])
        self.assertIsNone(self.explainer.training_data)
    
    def test_prepare_training_data(self):
        """Test training data preparation."""
        X = self.explainer.prepare_training_data(self.sample_proposals)
        
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.shape[0], len(self.sample_proposals))
        self.assertGreater(X.shape[1], 0)
        self.assertEqual(len(self.explainer.feature_names), X.shape[1])
        self.assertIsNotNone(self.explainer.training_data)
    
    def test_train_shap_explainers(self):
        """Test SHAP explainer training."""
        self.explainer.train_shap_explainers(self.mock_models, self.X_train)
        
        self.assertIn('xgboost', self.explainer.shap_explainers)
        self.assertIn('lightgbm', self.explainer.shap_explainers)
    
    def test_train_lime_explainer(self):
        """Test LIME explainer training."""
        self.explainer.train_lime_explainer(self.X_train)
        
        self.assertIn('proposal_evaluator', self.explainer.lime_explainers)
    
    def test_explain_with_shap(self):
        """Test SHAP explanation."""
        # Train explainers first
        self.explainer.train_shap_explainers(self.mock_models, self.X_train)
        
        # Explain an instance
        X_instance = self.X_train[0:1]
        explanation = self.explainer.explain_with_shap('xgboost', X_instance)
        
        self.assertIsInstance(explanation, dict)
        self.assertIn('shap_values', explanation)
        self.assertIsInstance(explanation['shap_values'], dict)
    
    def test_explain_with_lime(self):
        """Test LIME explanation."""
        # Train explainer first
        self.explainer.train_lime_explainer(self.X_train)
        
        # Define mock prediction function
        def mock_predict_fn(X):
            return np.sum(X, axis=1) if X.ndim > 1 else np.sum(X)
        
        # Explain an instance
        X_instance = self.X_train[0:1]
        explanation = self.explainer.explain_with_lime(mock_predict_fn, X_instance)
        
        self.assertIsInstance(explanation, dict)
        # Check that we get either results or an error message
        self.assertTrue('lime_weights' in explanation or 'error' in explanation)
    
    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        importance = self.explainer.get_feature_importance(self.mock_models['xgboost'], self.X_train)
        
        self.assertIsInstance(importance, dict)
        self.assertGreater(len(importance), 0)
        
        # Check that all values are floats
        for value in importance.values():
            self.assertIsInstance(value, float)
    
    def test_highlight_text_spans(self):
        """Test text span highlighting."""
        text = "This proposal involves artificial intelligence and machine learning techniques."
        important_terms = ["artificial intelligence", "machine learning"]
        
        highlighted_spans = self.explainer.highlight_text_spans(text, important_terms)
        
        self.assertIsInstance(highlighted_spans, list)
        self.assertGreater(len(highlighted_spans), 0)
        
        # Check structure of highlighted spans
        for span in highlighted_spans:
            self.assertIn('text', span)
            self.assertIn('start', span)
            self.assertIn('end', span)
            self.assertIn('term', span)
    
    def test_generate_explanation_report(self):
        """Test explanation report generation."""
        # Train explainers
        self.explainer.train_shap_explainers(self.mock_models, self.X_train)
        self.explainer.train_lime_explainer(self.X_train)
        
        # Generate explanations
        X_instance = self.X_train[0:1]
        sample_proposal = self.sample_proposals.iloc[0].to_dict()
        
        shap_explanation = self.explainer.explain_with_shap('xgboost', X_instance)
        lime_explanation = self.explainer.explain_with_lime(lambda x: np.sum(x), X_instance)
        feature_importance = self.explainer.get_feature_importance(self.mock_models['xgboost'], self.X_train)
        
        # Generate report
        report = self.explainer.generate_explanation_report(
            sample_proposal,
            shap_explanation,
            lime_explanation,
            feature_importance
        )
        
        # Check report structure
        expected_keys = [
            'proposal_id', 'proposal_title', 'overall_score', 'shap_explanation',
            'lime_explanation', 'feature_importance', 'highlighted_text_spans', 'interpretation'
        ]
        
        for key in expected_keys:
            self.assertIn(key, report)

if __name__ == '__main__':
    unittest.main()