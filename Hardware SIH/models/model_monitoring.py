import pandas as pd
import numpy as np
import json
import os
import warnings
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class ModelMonitor:
    """
    A class to monitor ML models and detect drift in R&D proposal evaluation system
    """
    
    def __init__(self, model_name: str = "proposal_evaluator"):
        """
        Initialize the Model Monitor
        
        Args:
            model_name (str): Name of the model to monitor
        """
        self.model_name = model_name
        self.monitoring_data_file = f"{model_name}_monitoring_data.json"
        self.drift_threshold = 0.05  # 5% threshold for drift detection
        self.performance_threshold = 0.7  # 70% threshold for performance alerts
        self.monitoring_data = []
        self.load_monitoring_data()
    
    def log_prediction(self, proposal_id: str, features: Dict[str, float], 
                      prediction: float, actual: Optional[float] = None,
                      human_feedback: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a prediction for monitoring
        
        Args:
            proposal_id (str): ID of the proposal
            features (Dict[str, float]): Feature values used for prediction
            prediction (float): Model prediction
            actual (Optional[float]): Actual outcome (if available)
            human_feedback (Optional[Dict[str, Any]]): Human feedback on prediction
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "proposal_id": proposal_id,
            "features": features,
            "prediction": prediction,
            "actual": actual,
            "human_feedback": human_feedback,
            "model_version": "1.0"  # In a real system, this would be dynamic
        }
        
        self.monitoring_data.append(entry)
        self.save_monitoring_data()
    
    def detect_feature_drift(self, current_data: pd.DataFrame, 
                           reference_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Detect drift in feature distributions
        
        Args:
            current_data (pd.DataFrame): Current feature data
            reference_data (pd.DataFrame): Reference feature data
            
        Returns:
            Dict[str, Dict[str, float]]: Drift detection results for each feature
        """
        drift_results = {}
        
        # Get common columns
        common_columns = list(set(current_data.columns) & set(reference_data.columns))
        
        for column in common_columns:
            try:
                # Statistical tests for drift detection
                # 1. Kolmogorov-Smirnov test for continuous features
                ks_statistic, ks_p_value = stats.ks_2samp(
                    reference_data[column].dropna(), 
                    current_data[column].dropna()
                )
                
                # 2. Population Stability Index (PSI) for binned features
                psi = self._calculate_psi(reference_data[column], current_data[column])
                
                # 3. Wasserstein distance for distribution similarity
                wasserstein_dist = stats.wasserstein_distance(
                    reference_data[column].dropna(), 
                    current_data[column].dropna()
                )
                
                # Determine if drift is detected
                drift_detected = (
                    ks_p_value < 0.05 or  # Statistical significance
                    psi > self.drift_threshold or  # PSI threshold
                    wasserstein_dist > self.drift_threshold  # Wasserstein threshold
                )
                
                drift_results[column] = {
                    "ks_statistic": float(ks_statistic),
                    "ks_p_value": float(ks_p_value),
                    "psi": float(psi),
                    "wasserstein_distance": float(wasserstein_dist),
                    "drift_detected": bool(drift_detected),
                    "drift_severity": self._classify_drift_severity(psi, wasserstein_dist)
                }
                
            except Exception as e:
                drift_results[column] = {
                    "error": str(e),
                    "drift_detected": False,
                    "drift_severity": "None"
                }
        
        return drift_results
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        Args:
            reference (pd.Series): Reference data
            current (pd.Series): Current data
            
        Returns:
            float: PSI value
        """
        # Create 10 bins
        bins = 10
        
        # Calculate percentages for reference data
        ref_hist, ref_bins = np.histogram(reference.dropna(), bins=bins)
        ref_percents = ref_hist / len(reference.dropna())
        
        # Calculate percentages for current data using same bins
        curr_hist, _ = np.histogram(current.dropna(), bins=ref_bins)
        curr_percents = curr_hist / len(current.dropna())
        
        # Avoid division by zero
        ref_percents = np.where(ref_percents == 0, 0.0001, ref_percents)
        curr_percents = np.where(curr_percents == 0, 0.0001, curr_percents)
        
        # Calculate PSI
        psi = np.sum((curr_percents - ref_percents) * np.log(curr_percents / ref_percents))
        
        return psi
    
    def _classify_drift_severity(self, psi: float, wasserstein: float) -> str:
        """
        Classify drift severity based on PSI and Wasserstein distance
        
        Args:
            psi (float): PSI value
            wasserstein (float): Wasserstein distance
            
        Returns:
            str: Drift severity classification
        """
        # PSI interpretation
        if psi >= 0.25:
            return "High"
        elif psi >= 0.1:
            return "Medium"
        elif psi >= 0.05:
            return "Low"
        else:
            return "None"
    
    def detect_prediction_drift(self, current_predictions: List[float], 
                              reference_predictions: List[float]) -> Dict[str, Any]:
        """
        Detect drift in prediction distributions
        
        Args:
            current_predictions (List[float]): Current predictions
            reference_predictions (List[float]): Reference predictions
            
        Returns:
            Dict[str, Any]: Prediction drift detection results
        """
        try:
            # Convert to numpy arrays
            current_array = np.array(current_predictions)
            reference_array = np.array(reference_predictions)
            
            # Statistical tests
            ks_statistic, ks_p_value = stats.ks_2samp(reference_array, current_array)
            
            # PSI calculation
            psi = self._calculate_psi(pd.Series(reference_array), pd.Series(current_array))
            
            # Wasserstein distance
            wasserstein_dist = stats.wasserstein_distance(reference_array, current_array)
            
            # Mean and std comparison
            ref_mean, ref_std = np.mean(reference_array), np.std(reference_array)
            curr_mean, curr_std = np.mean(current_array), np.std(current_array)
            
            mean_diff = abs(curr_mean - ref_mean) / (ref_std + 1e-8)  # Standardized difference
            std_ratio = curr_std / (ref_std + 1e-8)
            
            # Drift detection
            drift_detected = (
                ks_p_value < 0.05 or
                psi > self.drift_threshold or
                wasserstein_dist > self.drift_threshold or
                mean_diff > 0.1 or  # 10% standardized mean difference
                abs(std_ratio - 1) > 0.1  # 10% standard deviation change
            )
            
            return {
                "ks_statistic": float(ks_statistic),
                "ks_p_value": float(ks_p_value),
                "psi": float(psi),
                "wasserstein_distance": float(wasserstein_dist),
                "reference_mean": float(ref_mean),
                "reference_std": float(ref_std),
                "current_mean": float(curr_mean),
                "current_std": float(curr_std),
                "mean_difference": float(mean_diff),
                "std_ratio": float(std_ratio),
                "drift_detected": bool(drift_detected),
                "drift_severity": self._classify_drift_severity(psi, wasserstein_dist)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "drift_detected": False,
                "drift_severity": "None"
            }
    
    def evaluate_model_performance(self, y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """
        Evaluate model performance (for classification tasks)
        
        Args:
            y_true (List[float]): True labels
            y_pred (List[float]): Predicted labels
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        try:
            # Convert to binary classification (threshold at 0.5)
            y_true_binary = [1 if y >= 0.5 else 0 for y in y_true]
            y_pred_binary = [1 if y >= 0.5 else 0 for y in y_pred]
            
            # Calculate metrics
            accuracy = accuracy_score(y_true_binary, y_pred_binary)
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            
            # Performance alert
            performance_alert = accuracy < self.performance_threshold
            
            return {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "performance_alert": bool(performance_alert)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "performance_alert": False
            }
    
    def generate_monitoring_report(self, reference_data: pd.DataFrame = None,
                                 current_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Generate a comprehensive monitoring report
        
        Args:
            reference_data (pd.DataFrame): Reference data for comparison
            current_data (pd.DataFrame): Current data for comparison
            
        Returns:
            Dict[str, Any]: Monitoring report
        """
        report = {
            "model_name": self.model_name,
            "report_timestamp": datetime.now().isoformat(),
            "total_predictions": len(self.monitoring_data),
            "drift_detection": {},
            "performance_metrics": {},
            "alerts": []
        }
        
        # Add feature drift detection if data provided
        if reference_data is not None and current_data is not None:
            feature_drift = self.detect_feature_drift(current_data, reference_data)
            report["drift_detection"]["feature_drift"] = feature_drift
            
            # Check for alerts
            for feature, results in feature_drift.items():
                if results.get("drift_detected", False):
                    report["alerts"].append({
                        "type": "feature_drift",
                        "feature": feature,
                        "severity": results.get("drift_severity", "Unknown"),
                        "details": results
                    })
        
        # Add prediction drift detection if we have enough data
        if len(self.monitoring_data) > 10:  # Need at least 10 predictions
            # Extract predictions for drift detection
            predictions = [entry["prediction"] for entry in self.monitoring_data[-100:]]  # Last 100
            if len(predictions) > 10:
                # Compare with earlier predictions (if available)
                if len(self.monitoring_data) > 200:
                    reference_predictions = [entry["prediction"] for entry in self.monitoring_data[-200:-100]]
                    prediction_drift = self.detect_prediction_drift(predictions, reference_predictions)
                    report["drift_detection"]["prediction_drift"] = prediction_drift
                    
                    if prediction_drift.get("drift_detected", False):
                        report["alerts"].append({
                            "type": "prediction_drift",
                            "severity": prediction_drift.get("drift_severity", "Unknown"),
                            "details": prediction_drift
                        })
        
        # Calculate performance metrics if we have actual values
        actual_values = [entry["actual"] for entry in self.monitoring_data if entry["actual"] is not None]
        predicted_values = [entry["prediction"] for entry in self.monitoring_data if entry["actual"] is not None]
        
        if len(actual_values) > 5:  # Need at least 5 actual values
            performance_metrics = self.evaluate_model_performance(actual_values, predicted_values)
            report["performance_metrics"] = performance_metrics
            
            if performance_metrics.get("performance_alert", False):
                report["alerts"].append({
                    "type": "performance_degradation",
                    "severity": "High",
                    "details": performance_metrics
                })
        
        # Add human feedback analysis if available
        feedback_count = len([entry for entry in self.monitoring_data if entry["human_feedback"] is not None])
        if feedback_count > 0:
            report["human_feedback_count"] = feedback_count
            
            # Calculate agreement between model and human feedback
            agreement_count = 0
            total_feedback = 0
            
            for entry in self.monitoring_data:
                if entry["human_feedback"] is not None:
                    total_feedback += 1
                    human_accept = entry["human_feedback"].get("accept", True)
                    model_recommendation = entry["prediction"] >= 0.7  # Assuming 0.7 threshold for acceptance
                    
                    if human_accept == model_recommendation:
                        agreement_count += 1
            
            if total_feedback > 0:
                agreement_rate = agreement_count / total_feedback
                report["human_model_agreement"] = agreement_rate
                
                if agreement_rate < 0.7:  # Less than 70% agreement
                    report["alerts"].append({
                        "type": "low_human_agreement",
                        "severity": "Medium",
                        "details": {
                            "agreement_rate": agreement_rate,
                            "total_feedback": total_feedback,
                            "agreement_count": agreement_count
                        }
                    })
        
        return report
    
    def save_monitoring_data(self) -> None:
        """
        Save monitoring data to file
        """
        try:
            with open(self.monitoring_data_file, 'w') as f:
                json.dump(self.monitoring_data, f, indent=2)
        except Exception as e:
            warnings.warn(f"Could not save monitoring data: {e}")
    
    def load_monitoring_data(self) -> None:
        """
        Load monitoring data from file
        """
        if os.path.exists(self.monitoring_data_file):
            try:
                with open(self.monitoring_data_file, 'r') as f:
                    self.monitoring_data = json.load(f)
            except Exception as e:
                warnings.warn(f"Could not load monitoring data: {e}")
                self.monitoring_data = []
    
    def get_recent_predictions(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get recent predictions within specified hours
        
        Args:
            hours (int): Number of hours to look back
            
        Returns:
            List[Dict[str, Any]]: Recent predictions
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_predictions = []
        
        for entry in self.monitoring_data:
            entry_time = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
            if entry_time >= cutoff_time:
                recent_predictions.append(entry)
        
        return recent_predictions
    
    def trigger_retraining_alert(self) -> bool:
        """
        Determine if model retraining should be triggered based on monitoring data
        
        Returns:
            bool: True if retraining should be triggered
        """
        # Get monitoring report
        report = self.generate_monitoring_report()
        
        # Check for high-severity alerts
        high_severity_alerts = [
            alert for alert in report.get("alerts", []) 
            if alert.get("severity") in ["High", "Critical"]
        ]
        
        # Trigger retraining if we have high-severity alerts
        return len(high_severity_alerts) > 0

def create_sample_monitoring_data() -> List[Dict[str, Any]]:
    """
    Create sample monitoring data for demonstration
    
    Returns:
        List[Dict[str, Any]]: Sample monitoring data
    """
    sample_data = []
    
    # Generate sample data for the last 30 days
    for i in range(100):
        entry = {
            "timestamp": (datetime.now() - timedelta(days=i % 30, hours=i % 24)).isoformat(),
            "proposal_id": f"PROP{i:03d}",
            "features": {
                "novelty_score": np.random.beta(2, 2),
                "financial_score": np.random.beta(2, 2),
                "technical_score": np.random.beta(2, 2),
                "coal_relevance_score": np.random.beta(2, 2),
                "alignment_score": np.random.beta(2, 2),
                "clarity_score": np.random.beta(2, 2),
                "impact_score": np.random.beta(2, 2)
            },
            "prediction": np.random.beta(2, 2),
            "actual": np.random.beta(2, 2) if i % 3 == 0 else None,  # Every 3rd has actual value
            "human_feedback": {
                "accept": np.random.random() > 0.2,  # 80% acceptance rate
                "rating": np.random.uniform(1, 5),
                "comments": "Sample feedback"
            } if i % 5 == 0 else None,  # Every 5th has feedback
            "model_version": "1.0"
        }
        sample_data.append(entry)
    
    return sample_data

def main():
    """
    Main function to demonstrate the Model Monitor
    """
    print("Demonstrating Model Monitor for R&D Proposal Evaluation System")
    print("=" * 65)
    
    # Initialize monitor
    monitor = ModelMonitor("proposal_evaluator")
    print("✓ Model Monitor initialized")
    
    # Create and load sample data
    sample_data = create_sample_monitoring_data()
    monitor.monitoring_data = sample_data
    monitor.save_monitoring_data()
    print(f"✓ Sample monitoring data loaded ({len(sample_data)} entries)")
    
    # Log a new prediction
    monitor.log_prediction(
        proposal_id="NEW001",
        features={
            "novelty_score": 0.85,
            "financial_score": 0.75,
            "technical_score": 0.90,
            "coal_relevance_score": 0.95,
            "alignment_score": 0.88,
            "clarity_score": 0.82,
            "impact_score": 0.90
        },
        prediction=0.87,
        actual=0.85,
        human_feedback={
            "accept": True,
            "rating": 4.5,
            "comments": "Excellent proposal with strong technical approach"
        }
    )
    print("✓ New prediction logged")
    
    # Generate monitoring report
    print("\nGenerating monitoring report...")
    report = monitor.generate_monitoring_report()
    
    print(f"Model: {report['model_name']}")
    print(f"Total Predictions: {report['total_predictions']}")
    print(f"Human Feedback Count: {report.get('human_feedback_count', 0)}")
    print(f"Human-Model Agreement: {report.get('human_model_agreement', 0):.2%}")
    
    # Show alerts if any
    alerts = report.get("alerts", [])
    if alerts:
        print(f"\nAlerts Detected ({len(alerts)}):")
        for i, alert in enumerate(alerts, 1):
            print(f"  {i}. {alert['type']} - {alert['severity']} severity")
    else:
        print("\nNo alerts detected")
    
    # Check if retraining is needed
    retrain_needed = monitor.trigger_retraining_alert()
    print(f"\nRetraining Alert: {'Yes' if retrain_needed else 'No'}")
    
    # Show recent predictions
    recent_predictions = monitor.get_recent_predictions(hours=24)
    print(f"\nRecent Predictions (24h): {len(recent_predictions)}")
    
    # Demonstrate feature drift detection
    print("\nDemonstrating feature drift detection...")
    
    # Create sample reference and current data
    np.random.seed(42)  # For reproducible results
    
    reference_data = pd.DataFrame({
        'novelty_score': np.random.beta(2, 2, 1000),
        'financial_score': np.random.beta(2, 2, 1000),
        'technical_score': np.random.beta(2, 2, 1000)
    })
    
    # Current data with some drift
    current_data = pd.DataFrame({
        'novelty_score': np.random.beta(2.5, 1.5, 1000),  # Drifted
        'financial_score': np.random.beta(2, 2, 1000),    # No drift
        'technical_score': np.random.beta(1.5, 2.5, 1000) # Drifted
    })
    
    feature_drift = monitor.detect_feature_drift(current_data, reference_data)
    
    print("Feature Drift Detection Results:")
    for feature, results in feature_drift.items():
        drift_detected = results.get('drift_detected', False)
        severity = results.get('drift_severity', 'Unknown')
        print(f"  {feature}: {'DRIFT' if drift_detected else 'NO DRIFT'} ({severity} severity)")
    
    print("\n✓ Model Monitor demonstration completed successfully")

if __name__ == "__main__":
    main()