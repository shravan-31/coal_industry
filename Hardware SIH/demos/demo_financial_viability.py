from financial_viability import FinancialViabilityEvaluator, create_sample_budget_data
import pandas as pd

def main():
    print("Demonstrating Financial Viability Evaluator")
    print("=" * 50)
    
    # Create evaluator
    evaluator = FinancialViabilityEvaluator()
    print("✓ Financial Viability Evaluator initialized")
    
    # Create sample data
    sample_data = create_sample_budget_data()
    historical_df = pd.DataFrame(sample_data[:-1])  # Use first two as historical
    new_proposal = sample_data[-1]  # Use last as new proposal
    
    print(f"✓ Sample data created with {len(sample_data)} proposals")
    
    # Train anomaly detector
    try:
        evaluator.train_anomaly_detector(historical_df)
        print("✓ Anomaly detector trained")
    except Exception as e:
        print(f"Warning: Could not train anomaly detector: {e}")
    
    # Evaluate a proposal
    proposal = sample_data[0]
    print(f"\nEvaluating proposal: {proposal['title']}")
    result = evaluator.evaluate_proposal(proposal)
    
    print(f"Financial Score: {result['financial_score']:.4f}")
    print(f"Actual Budget: ${result['actual_budget']:,.2f}")
    print(f"Expected Budget: ${result['expected_budget']:,.2f}")
    print(f"Budget Z-Score: {result['budget_zscore']:.2f}")
    
    if result['guideline_validation']['violations']:
        print("\nGuideline Violations:")
        for violation in result['guideline_validation']['violations']:
            print(f"  - {violation['description']} ({violation['severity']})")
    else:
        print("\n✓ No guideline violations found")
    
    if result['guideline_validation']['suggestions']:
        print("\nRecommendations:")
        for suggestion in result['guideline_validation']['suggestions']:
            print(f"  - {suggestion}")
    
    print("\n✓ Financial Viability Evaluator demonstration completed successfully")

if __name__ == "__main__":
    main()