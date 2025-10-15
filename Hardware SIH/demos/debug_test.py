from proposal_evaluator import RDPEvaluator, create_sample_data
import pandas as pd

def debug_test():
    # Create sample data
    create_sample_data()
    
    # Initialize evaluator
    evaluator = RDPEvaluator()
    
    # Evaluate proposals
    print("Evaluating proposals...")
    results = evaluator.evaluate_proposals(
        'sample_past_proposals.csv', 
        'sample_new_proposals.csv',
        include_feasibility=True
    )
    
    # Print column names
    print("Column names in results:")
    print(results.columns.tolist())
    
    # Print first few rows
    print("\nFirst few rows:")
    print(results.head())

if __name__ == "__main__":
    debug_test()