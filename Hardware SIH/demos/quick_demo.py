"""
Quick Demo of R&D Proposal Evaluation System Components
"""
import pandas as pd
from enhanced_evaluator import create_sample_data

def quick_demo():
    """Quick demonstration of system components"""
    print("Quick Demo of R&D Proposal Evaluation System")
    print("=" * 50)
    
    # Create sample data
    print("1. Creating sample data...")
    create_sample_data()
    print("   ✓ Sample data created")
    
    # Show sample data
    print("\n2. Sample proposals:")
    try:
        df = pd.read_csv('sample_new_proposals.csv')
        for i, row in df.iterrows():
            print(f"   {i+1}. {row['Title']}")
    except Exception as e:
        print(f"   Error reading sample data: {e}")
    
    # Test novelty detector
    print("\n3. Testing novelty detection...")
    try:
        from novelty_detector import create_proposal_kb_from_csv
        create_proposal_kb_from_csv(
            'sample_past_proposals.csv', 
            'demo_proposals.index', 
            'demo_proposals_metadata.json'
        )
        print("   ✓ Novelty detection ready")
    except Exception as e:
        print(f"   Error with novelty detection: {e}")
    
    # Test technical feasibility
    print("\n4. Testing technical feasibility module...")
    try:
        from technical_feasibility import TechnicalFeasibilityEvaluator
        evaluator = TechnicalFeasibilityEvaluator()
        print("   ✓ Technical feasibility evaluator ready")
    except Exception as e:
        print(f"   Error with technical feasibility: {e}")
    
    # Test financial viability
    print("\n5. Testing financial viability module...")
    try:
        from financial_viability import FinancialViabilityEvaluator
        evaluator = FinancialViabilityEvaluator()
        print("   ✓ Financial viability evaluator ready")
    except Exception as e:
        print(f"   Error with financial viability: {e}")
    
    print("\n" + "=" * 50)
    print("Quick demo completed!")
    print("\nSystem components are ready for use.")
    print("Run 'streamlit run app_hil.py' to start the web interface.")

if __name__ == "__main__":
    quick_demo()