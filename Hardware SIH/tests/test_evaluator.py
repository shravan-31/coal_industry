from proposal_evaluator import RDPEvaluator, create_sample_data

def test_evaluator():
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
    
    # Save results
    evaluator.save_results(results, 'evaluated_proposals.csv')
    
    # Display top proposals
    evaluator.display_top_proposals(results, top_n=5)
    
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    test_evaluator()