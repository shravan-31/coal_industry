from model_monitoring import ModelMonitor
import pandas as pd
import numpy as np

def main():
    print("Demonstrating Model Monitor for R&D Proposal Evaluation System")
    print("=" * 65)
    
    # Initialize monitor
    monitor = ModelMonitor("proposal_evaluator")
    print("✓ Model Monitor initialized")
    
    # Log several predictions
    sample_predictions = [
        {
            "proposal_id": "PROP001",
            "features": {
                "novelty_score": 0.85,
                "financial_score": 0.75,
                "technical_score": 0.90,
                "coal_relevance_score": 0.95,
                "alignment_score": 0.88,
                "clarity_score": 0.82,
                "impact_score": 0.90
            },
            "prediction": 0.87,
            "actual": 0.85,
            "human_feedback": {
                "accept": True,
                "rating": 4.5,
                "comments": "Excellent proposal with strong technical approach"
            }
        },
        {
            "proposal_id": "PROP002",
            "features": {
                "novelty_score": 0.65,
                "financial_score": 0.85,
                "technical_score": 0.70,
                "coal_relevance_score": 0.80,
                "alignment_score": 0.75,
                "clarity_score": 0.68,
                "impact_score": 0.72
            },
            "prediction": 0.74,
            "actual": 0.70,
            "human_feedback": {
                "accept": True,
                "rating": 3.8,
                "comments": "Good proposal but could be improved"
            }
        },
        {
            "proposal_id": "PROP003",
            "features": {
                "novelty_score": 0.45,
                "financial_score": 0.55,
                "technical_score": 0.50,
                "coal_relevance_score": 0.60,
                "alignment_score": 0.55,
                "clarity_score": 0.48,
                "impact_score": 0.52
            },
            "prediction": 0.53,
            "actual": 0.45,
            "human_feedback": {
                "accept": False,
                "rating": 2.5,
                "comments": "Proposal needs significant improvement"
            }
        }
    ]
    
    # Log predictions
    for pred in sample_predictions:
        monitor.log_prediction(
            proposal_id=pred["proposal_id"],
            features=pred["features"],
            prediction=pred["prediction"],
            actual=pred["actual"],
            human_feedback=pred["human_feedback"]
        )
    
    print(f"✓ {len(sample_predictions)} predictions logged")
    
    # Generate monitoring report
    print("\nGenerating monitoring report...")
    report = monitor.generate_monitoring_report()
    
    print(f"Model: {report['model_name']}")
    print(f"Report Time: {report['report_timestamp']}")
    print(f"Total Predictions: {report['total_predictions']}")
    print(f"Human Feedback Count: {report.get('human_feedback_count', 0)}")
    print(f"Human-Model Agreement: {report.get('human_model_agreement', 0):.2%}")
    
    # Show performance metrics
    perf_metrics = report.get('performance_metrics', {})
    if perf_metrics:
        print("\nPerformance Metrics:")
        print(f"  Accuracy: {perf_metrics.get('accuracy', 0):.4f}")
        print(f"  Precision: {perf_metrics.get('precision', 0):.4f}")
        print(f"  Recall: {perf_metrics.get('recall', 0):.4f}")
        print(f"  F1-Score: {perf_metrics.get('f1_score', 0):.4f}")
    
    # Show alerts if any
    alerts = report.get("alerts", [])
    if alerts:
        print(f"\nAlerts Detected ({len(alerts)}):")
        for i, alert in enumerate(alerts, 1):
            print(f"  {i}. {alert['type']} - {alert['severity']} severity")
    else:
        print("\n✓ No alerts detected")
    
    # Check if retraining is needed
    retrain_needed = monitor.trigger_retraining_alert()
    print(f"\nRetraining Alert: {'Yes - Model performance may be degrading' if retrain_needed else 'No - Model performing well'}")
    
    # Demonstrate drift detection
    print("\nDemonstrating drift detection...")
    
    # Create reference data (normal distribution)
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'novelty_score': np.random.beta(2, 2, 1000),
        'financial_score': np.random.beta(2, 2, 1000),
        'technical_score': np.random.beta(2, 2, 1000)
    })
    
    # Create current data with some drift
    current_data = pd.DataFrame({
        'novelty_score': np.random.beta(2.5, 1.5, 1000),  # Drifted toward higher values
        'financial_score': np.random.beta(2, 2, 1000),    # No drift
        'technical_score': np.random.beta(1.5, 2.5, 1000) # Drifted toward lower values
    })
    
    # Detect feature drift
    feature_drift = monitor.detect_feature_drift(current_data, reference_data)
    
    print("Feature Drift Detection Results:")
    for feature, results in feature_drift.items():
        drift_detected = results.get('drift_detected', False)
        severity = results.get('drift_severity', 'Unknown')
        psi = results.get('psi', 0)
        print(f"  {feature}: {'DRIFT DETECTED' if drift_detected else 'NO DRIFT'} ({severity} severity, PSI: {psi:.4f})")
    
    # Detect prediction drift
    reference_predictions = np.random.beta(2, 2, 1000).tolist()
    current_predictions = np.random.beta(2.5, 1.5, 1000).tolist()  # Shifted toward higher values
    
    prediction_drift = monitor.detect_prediction_drift(current_predictions, reference_predictions)
    
    print("\nPrediction Drift Detection Results:")
    drift_detected = prediction_drift.get('drift_detected', False)
    severity = prediction_drift.get('drift_severity', 'Unknown')
    psi = prediction_drift.get('psi', 0)
    print(f"  Predictions: {'DRIFT DETECTED' if drift_detected else 'NO DRIFT'} ({severity} severity, PSI: {psi:.4f})")
    
    print("\n✓ Model Monitor demonstration completed successfully")

if __name__ == "__main__":
    main()