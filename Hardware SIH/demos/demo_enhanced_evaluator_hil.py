from enhanced_evaluator_hil import EnhancedRDPEvaluatorHIL, create_sample_data
import pandas as pd

def main():
    print("Demonstrating Enhanced R&D Proposal Evaluator with Human-in-the-Loop")
    print("=" * 70)
    
    # Create sample data
    create_sample_data()
    print("✓ Sample data created")
    
    # Initialize evaluator
    evaluator = EnhancedRDPEvaluatorHIL()
    print("✓ Enhanced Evaluator initialized")
    
    # Show default weights
    default_weights = evaluator.get_weights()
    print(f"\nDefault Weights: {default_weights}")
    
    # Adjust weights to emphasize novelty
    new_weights = {
        'novelty': 0.30,
        'financial': 0.10,
        'technical': 0.20,
        'coal_relevance': 0.10,
        'alignment': 0.10,
        'clarity': 0.10,
        'impact': 0.10
    }
    
    if evaluator.set_weights(new_weights):
        print(f"✓ New weights set: {evaluator.get_weights()}")
    else:
        print("✗ Failed to set weights")
    
    # Evaluate proposals
    print("\nEvaluating proposals with custom weights...")
    results = evaluator.evaluate_proposals(
        'sample_past_proposals.csv',
        'sample_new_proposals.csv'
    )
    print("✓ Evaluation completed")
    
    # Display results
    print(f"\nEvaluation Results (using custom weights):")
    print("=" * 50)
    for index, proposal in results.iterrows():
        print(f"Proposal ID: {proposal['Proposal_ID']}")
        print(f"Title: {proposal['Title']}")
        print(f"Overall Score: {proposal['Overall_Score']:.2f}")
        print(f"Recommendation: {proposal['Recommendation']}")
        print("-" * 30)
    
    # Demonstrate feedback collection
    print("\nDemonstrating feedback collection...")
    feedback = {
        "comments": "Excellent proposal with strong technical approach",
        "rating": 4.8,
        "accept": True
    }
    
    evaluator.collect_human_feedback(
        proposal_id="N001",
        reviewer_id="REV001",
        feedback=feedback
    )
    print("✓ Feedback collected for proposal N001")
    
    # Retrieve feedback
    retrieved_feedback = evaluator.get_feedback_for_proposal("N001")
    print(f"✓ Retrieved {len(retrieved_feedback)} feedback entries for proposal N001")
    
    # Show feedback details
    if retrieved_feedback:
        fb = retrieved_feedback[0]
        print(f"  Reviewer: {fb['reviewer_id']}")
        print(f"  Comments: {fb['feedback']['comments']}")
        print(f"  Rating: {fb['feedback']['rating']}")
    
    # Demonstrate weight adjustment based on feedback
    print("\nDemonstrating weight adjustment based on feedback...")
    adjusted_weights = evaluator.adjust_weights_from_feedback()
    print(f"Suggested weights from feedback: {adjusted_weights}")
    
    print("\n✓ Enhanced Evaluator demonstration completed successfully")

if __name__ == "__main__":
    main()