from technical_feasibility import TechnicalFeasibilityEvaluator, create_sample_training_data
import pandas as pd

def main():
    print("Demonstrating Technical Feasibility Evaluator")
    print("=" * 50)
    
    # Create evaluator
    evaluator = TechnicalFeasibilityEvaluator()
    print("✓ Technical Feasibility Evaluator initialized")
    
    # Create sample data
    df, labels = create_sample_training_data()
    print(f"✓ Sample training data created with {len(df)} proposals")
    
    # Extract features
    numerical_features = evaluator.extract_numerical_features(df)
    print(f"✓ Numerical features extracted: {numerical_features.shape}")
    
    # Test proposal evaluation
    sample_proposal = {
        'Title': 'AI for Mine Safety Monitoring',
        'Abstract': 'Using artificial intelligence to monitor and predict safety hazards in coal mines to prevent accidents with a detailed technical approach.',
        'Funding_Requested': 80000
    }
    
    result = evaluator.evaluate_proposal(sample_proposal)
    print("\nSample Proposal Evaluation:")
    print(f"Title: {sample_proposal['Title']}")
    print(f"Feasibility Score: {result['feasibility_score']:.4f}")
    print(f"Team Experience: {result['team_experience_score']:.4f}")
    print(f"Budget Realism: {result['budget_realism_score']:.4f}")
    print(f"Methodology Completeness: {result['methodology_completeness_score']:.4f}")
    print(f"BERT Analysis: {result['bert_analysis_score']:.4f}")
    
    print("\n✓ Technical Feasibility Evaluator demonstration completed successfully")

if __name__ == "__main__":
    main()