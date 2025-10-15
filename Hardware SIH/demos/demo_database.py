from database import ProposalDatabase, create_sample_database_data

def main():
    print("Demonstrating Proposal Database for R&D Evaluation System")
    print("=" * 58)
    
    # Initialize database
    db = ProposalDatabase("demo_proposals.db")
    print("✓ Database initialized")
    
    # Create sample data
    create_sample_database_data(db)
    print("✓ Sample data created")
    
    # Demonstrate database operations
    print("\nDatabase Operations Demo:")
    
    # 1. Get proposal
    print("\n1. Get Proposal:")
    proposal = db.get_proposal('PROP001')
    if proposal:
        print(f"   Title: {proposal['title']}")
        print(f"   PI: {proposal['pi_name']}")
        print(f"   Funding: ${proposal['funding_requested']:,.2f}")
        print(f"   Sections: {list(proposal['sections'].keys())}")
    
    # 2. Get evaluation scores
    print("\n2. Get Evaluation Scores:")
    scores = db.get_evaluation_scores('PROP001')
    if scores:
        score = scores[0]  # Most recent
        print(f"   Overall Score: {score['overall_score']}")
        print(f"   Novelty: {score['novelty_score']}")
        print(f"   Technical: {score['technical_score']}")
        print(f"   Detailed scores: {len(score['detailed_scores'])} parameters")
    
    # 3. Get feedback
    print("\n3. Get Feedback:")
    feedback_list = db.get_feedback('PROP001')
    if feedback_list:
        feedback = feedback_list[0]  # Most recent
        print(f"   Reviewer: {feedback['reviewer_id']}")
        print(f"   Rating: {feedback['rating']}")
        print(f"   Accept: {feedback['accept']}")
        print(f"   Comments: {feedback['comments'][:50]}...")
    
    # 4. Get reviewer
    print("\n4. Get Reviewer:")
    reviewer = db.get_reviewer('REV001')
    if reviewer:
        print(f"   Name: {reviewer['name']}")
        print(f"   Email: {reviewer['email']}")
        print(f"   Expertise: {reviewer['expertise']}")
        print(f"   Active: {reviewer['is_active']}")
    
    # 5. Get audit log
    print("\n5. Get Audit Log:")
    audit_log = db.get_audit_log('PROP001', limit=5)
    print(f"   Recent actions: {len(audit_log)}")
    if audit_log:
        latest_action = audit_log[0]
        print(f"   Latest: {latest_action['action_type']} by {latest_action['user_id']}")
    
    # 6. Get database stats
    print("\n6. Database Statistics:")
    stats = db.get_database_stats()
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    # 7. Export to DataFrame
    print("\n7. Export to DataFrame:")
    try:
        import pandas as pd
        proposals_df = db.export_to_dataframe('proposals')
        print(f"   Proposals DataFrame shape: {proposals_df.shape}")
        
        evaluations_df = db.export_to_dataframe('evaluation_scores')
        print(f"   Evaluations DataFrame shape: {evaluations_df.shape}")
    except ImportError:
        print("   Pandas not available for DataFrame export")
    
    print("\n" + "=" * 58)
    print("✓ Database demonstration completed successfully")
    print(f"\nNote: Demo database saved as 'demo_proposals.db'")

if __name__ == "__main__":
    main()