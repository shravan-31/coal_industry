import unittest
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from model_monitoring import ModelMonitor

class TestModelMonitor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.monitor = ModelMonitor("test_model")
        self.test_file = "test_model_monitoring_data.json"
        
        # Remove test file if it exists
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Remove test file
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_initialization(self):
        """Test that the monitor initializes correctly."""
        self.assertEqual(self.monitor.model_name, "test_model")
        self.assertEqual(self.monitor.monitoring_data, [])
        self.assertEqual(self.monitor.drift_threshold, 0.05)
    
    def test_log_prediction(self):
        """Test logging a prediction."""
        features = {
            "novelty_score": 0.85,
            "financial_score": 0.75,
            "technical_score": 0.90
        }
        
        self.monitor.log_prediction(
            proposal_id="TEST001",
            features=features,
            prediction=0.87
        )
        
        self.assertEqual(len(self.monitor.monitoring_data), 1)
        entry = self.monitor.monitoring_data[0]
        self.assertEqual(entry["proposal_id"], "TEST001")
        self.assertEqual(entry["prediction"], 0.87)
        self.assertEqual(entry["features"], features)
    
    def test_save_and_load_monitoring_data(self):
        """Test saving and loading monitoring data."""
        # Add some data
        self.monitor.log_prediction("TEST001", {"score": 0.8}, 0.8)
        self.monitor.log_prediction("TEST002", {"score": 0.9}, 0.9)
        
        # Save data
        self.monitor.save_monitoring_data()
        
        # Create new monitor instance to test loading
        new_monitor = ModelMonitor("test_model")
        
        # Check that data was loaded
        self.assertEqual(len(new_monitor.monitoring_data), 2)
        self.assertEqual(new_monitor.monitoring_data[0]["proposal_id"], "TEST001")
    
    def test_calculate_psi(self):
        """Test PSI calculation."""
        # Create sample data
        reference = pd.Series([1, 2, 3, 4, 5] * 20)  # 100 values
        current = pd.Series([1, 2, 3, 4, 5] * 20)    # Same distribution
        
        psi = self.monitor._calculate_psi(reference, current)
        
        # PSI should be close to 0 for same distributions
        self.assertLess(abs(psi), 0.01)
    
    def test_classify_drift_severity(self):
        """Test drift severity classification."""
        # Test different PSI values
        self.assertEqual(self.monitor._classify_drift_severity(0.30, 0.1), "High")
        self.assertEqual(self.monitor._classify_drift_severity(0.15, 0.1), "Medium")
        self.assertEqual(self.monitor._classify_drift_severity(0.07, 0.1), "Low")
        self.assertEqual(self.monitor._classify_drift_severity(0.02, 0.1), "None")
    
    def test_detect_feature_drift(self):
        """Test feature drift detection."""
        # Create sample data
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(0, 1, 1000)
        })
        
        # Current data with no drift
        current_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(0, 1, 1000)
        })
        
        drift_results = self.monitor.detect_feature_drift(current_data, reference_data)
        
        self.assertIsInstance(drift_results, dict)
        self.assertIn('feature1', drift_results)
        self.assertIn('feature2', drift_results)
        
        # Check structure of results
        for feature, results in drift_results.items():
            self.assertIn('ks_statistic', results)
            self.assertIn('ks_p_value', results)
            self.assertIn('psi', results)
            self.assertIn('drift_detected', results)
    
    def test_detect_prediction_drift(self):
        """Test prediction drift detection."""
        # Create sample predictions
        np.random.seed(42)
        reference_predictions = np.random.beta(2, 2, 1000).tolist()
        current_predictions = np.random.beta(2, 2, 1000).tolist()
        
        drift_results = self.monitor.detect_prediction_drift(current_predictions, reference_predictions)
        
        self.assertIsInstance(drift_results, dict)
        self.assertIn('ks_statistic', drift_results)
        self.assertIn('ks_p_value', drift_results)
        self.assertIn('psi', drift_results)
        self.assertIn('drift_detected', drift_results)
    
    def test_evaluate_model_performance(self):
        """Test model performance evaluation."""
        # Create sample true and predicted values
        y_true = [0.9, 0.8, 0.1, 0.2, 0.7, 0.6, 0.3, 0.4]
        y_pred = [0.8, 0.7, 0.2, 0.1, 0.6, 0.5, 0.4, 0.3]
        
        performance_metrics = self.monitor.evaluate_model_performance(y_true, y_pred)
        
        self.assertIsInstance(performance_metrics, dict)
        self.assertIn('accuracy', performance_metrics)
        self.assertIn('precision', performance_metrics)
        self.assertIn('recall', performance_metrics)
        self.assertIn('f1_score', performance_metrics)
    
    def test_generate_monitoring_report(self):
        """Test generating monitoring report."""
        # Add sample data
        self.monitor.log_prediction(
            proposal_id="TEST001",
            features={"score": 0.8},
            prediction=0.8,
            actual=0.75
        )
        
        self.monitor.log_prediction(
            proposal_id="TEST002",
            features={"score": 0.9},
            prediction=0.9,
            human_feedback={"accept": True, "rating": 4.5}
        )
        
        # Generate report
        report = self.monitor.generate_monitoring_report()
        
        self.assertIsInstance(report, dict)
        self.assertEqual(report["model_name"], "test_model")
        self.assertEqual(report["total_predictions"], 2)
        self.assertIn("performance_metrics", report)
        self.assertIn("alerts", report)
    
    def test_get_recent_predictions(self):
        """Test getting recent predictions."""
        # Add old prediction (30 hours ago)
        old_entry = {
            "timestamp": (datetime.now() - timedelta(hours=30)).isoformat(),
            "proposal_id": "OLD001",
            "features": {"score": 0.5},
            "prediction": 0.5
        }
        
        # Add recent prediction (1 hour ago)
        recent_entry = {
            "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
            "proposal_id": "NEW001",
            "features": {"score": 0.8},
            "prediction": 0.8
        }
        
        self.monitor.monitoring_data = [old_entry, recent_entry]
        
        # Get recent predictions (last 24 hours)
        recent_predictions = self.monitor.get_recent_predictions(hours=24)
        
        self.assertEqual(len(recent_predictions), 1)
        self.assertEqual(recent_predictions[0]["proposal_id"], "NEW001")
    
    def test_trigger_retraining_alert(self):
        """Test retraining alert trigger."""
        # Add sample data without alerts
        self.monitor.log_prediction("TEST001", {"score": 0.8}, 0.8)
        
        # Check retraining alert
        retrain_needed = self.monitor.trigger_retraining_alert()
        self.assertIsInstance(retrain_needed, bool)

if __name__ == '__main__':
    unittest.main()