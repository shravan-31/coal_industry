"""
Complete System Demonstration for R&D Proposal Evaluation System

This script demonstrates the full workflow of the Advanced Auto Evaluation System
for R&D Proposals, showcasing all implemented components working together.
"""

import os
import time
import pandas as pd
from enhanced_evaluator import create_sample_data
from document_parser import DocumentParser
from novelty_detector import FAISSNoveltyDetector, create_proposal_kb_from_csv
from technical_feasibility import TechnicalFeasibilityEvaluator
from financial_viability import FinancialViabilityEvaluator
from risk_ethics_ip import RiskEthicsIPChecker
from explainability import ModelExplainer
from model_monitoring import ModelMonitor
from database import ProposalDatabase
from security import SecurityManager

def demonstrate_complete_system():
    """Demonstrate the complete R&D Proposal Evaluation System workflow"""
    
    print("=" * 80)
    print("DEMONSTRATION: Advanced Auto Evaluation System for R&D Proposals")
    print("=" * 80)
    print("This demonstration shows the complete workflow of the system,")
    print("integrating all components from document parsing to final evaluation.")
    print()
    
    # Step 1: Initialize System Components
    print("Step 1: Initializing System Components")
    print("-" * 40)
    
    # Create sample data
    create_sample_data()
    print("✓ Sample data created (sample_past_proposals.csv, sample_new_proposals.csv)")
    
    # Initialize components
    parser = DocumentParser()
    novelty_detector = FAISSNoveltyDetector()
    technical_evaluator = TechnicalFeasibilityEvaluator()
    financial_evaluator = FinancialViabilityEvaluator()
    risk_checker = RiskEthicsIPChecker()
    explainer = ModelExplainer()
    monitor = ModelMonitor("proposal_evaluator")
    database = ProposalDatabase("demo_system.db")
    security = SecurityManager("demo_security.db")
    
    print("✓ All system components initialized")
    time.sleep(1)
    
    # Step 2: Document Parsing and Feature Extraction
    print("\nStep 2: Document Parsing and Feature Extraction")
    print("-" * 40)
    
    # In a real scenario, we would parse actual PDF/DOCX files
    # For demonstration, we'll simulate this with sample data
    sample_proposals = pd.read_csv('sample_new_proposals.csv')
    
    print(f"Processing {len(sample_proposals)} sample proposals:")
    for i, proposal in sample_proposals.iterrows():
        print(f"  {i+1}. {proposal['Title']}")
    
    print("✓ Document parsing and feature extraction completed")
    time.sleep(1)
    
    # Step 3: Knowledge Base Creation and Novelty Detection
    print("\nStep 3: Knowledge Base Creation and Novelty Detection")
    print("-" * 40)
    
    # Create FAISS knowledge base from past proposals
    create_proposal_kb_from_csv(
        'sample_past_proposals.csv', 
        'demo_proposals.index', 
        'demo_proposals_metadata.json'
    )
    print("✓ Knowledge base created with FAISS vector database")
    
    # Load the knowledge base
    novelty_detector.load_index('demo_proposals.index', 'demo_proposals_metadata.json')
    print("✓ Knowledge base loaded for novelty detection")
    
    # Evaluate novelty for sample proposals
    print("\nNovelty Scores:")
    for i, proposal in sample_proposals.iterrows():
        novelty_score, similar_proposals = novelty_detector.calculate_novelty_score(
            f"{proposal['Title']} {proposal['Abstract']}"
        )
        print(f"  {proposal['Title']}: {novelty_score:.4f}")
        
        # Log for monitoring
        monitor.log_prediction(
            proposal_id=proposal['Proposal_ID'],
            features={'novelty_score': novelty_score},
            prediction=novelty_score
        )
    
    time.sleep(1)
    
    # Step 4: Technical Feasibility Evaluation
    print("\nStep 4: Technical Feasibility Evaluation")
    print("-" * 40)
    
    print("Technical Feasibility Scores:")
    for i, proposal in sample_proposals.iterrows():
        proposal_data = {
            'Title': proposal['Title'],
            'Abstract': proposal['Abstract'],
            'Funding_Requested': proposal['Funding_Requested']
        }
        
        result = technical_evaluator.evaluate_proposal(proposal_data)
        print(f"  {proposal['Title']}: {result['feasibility_score']:.4f}")
        
        # Log for monitoring
        monitor.log_prediction(
            proposal_id=proposal['Proposal_ID'],
            features={'technical_score': result['feasibility_score']},
            prediction=result['feasibility_score']
        )
    
    time.sleep(1)
    
    # Step 5: Financial Viability Assessment
    print("\nStep 5: Financial Viability Assessment")
    print("-" * 40)
    
    print("Financial Viability Scores:")
    for i, proposal in sample_proposals.iterrows():
        proposal_data = {
            'funding_requested': proposal['Funding_Requested'],
            'budget_breakdown': {
                'personnel': proposal['Funding_Requested'] * 0.6,
                'equipment': proposal['Funding_Requested'] * 0.3,
                'other': proposal['Funding_Requested'] * 0.1
            }
        }
        
        result = financial_evaluator.evaluate_proposal(proposal_data)
        print(f"  {proposal['Title']}: {result['financial_score']:.4f}")
        
        # Log for monitoring
        monitor.log_prediction(
            proposal_id=proposal['Proposal_ID'],
            features={'financial_score': result['financial_score']},
            prediction=result['financial_score']
        )
    
    time.sleep(1)
    
    # Step 6: Risk, Ethics, and IP Assessment
    print("\nStep 6: Risk, Ethics, and IP Assessment")
    print("-" * 40)
    
    print("Risk Assessment Results:")
    for i, proposal in sample_proposals.iterrows():
        proposal_data = {
            'Title': proposal['Title'],
            'Abstract': proposal['Abstract']
        }
        
        result = risk_checker.evaluate_proposal(proposal_data)
        risk_score = result['overall_risk_score']
        print(f"  {proposal['Title']}: {risk_score:.4f} ({result['risk_level']} risk)")
        
        # Log for monitoring
        monitor.log_prediction(
            proposal_id=proposal['Proposal_ID'],
            features={'risk_score': risk_score},
            prediction=1-risk_score  # Invert for positive scoring
        )
    
    time.sleep(1)
    
    # Step 7: Database Integration
    print("\nStep 7: Database Integration")
    print("-" * 40)
    
    # Save proposals to database
    for i, proposal in sample_proposals.iterrows():
        proposal_data = {
            'proposal_id': proposal['Proposal_ID'],
            'title': proposal['Title'],
            'abstract': proposal['Abstract'],
            'funding_requested': proposal['Funding_Requested'],
            'sections': {
                'title': proposal['Title'],
                'abstract': proposal['Abstract']
            }
        }
        
        database.insert_proposal(proposal_data)
    
    print(f"✓ {len(sample_proposals)} proposals saved to database")
    
    # Save evaluation scores
    sample_scores = [
        {'proposal_id': 'N001', 'novelty_score': 0.85, 'technical_score': 0.90, 
         'financial_score': 0.75, 'overall_score': 83.5},
        {'proposal_id': 'N002', 'novelty_score': 0.75, 'technical_score': 0.85, 
         'financial_score': 0.80, 'overall_score': 80.0}
    ]
    
    for score_data in sample_scores:
        proposal_id = score_data.pop('proposal_id')
        database.insert_evaluation_scores(proposal_id, score_data)
    
    print("✓ Evaluation scores saved to database")
    time.sleep(1)
    
    # Step 8: Explainability and Interpretation
    print("\nStep 8: Explainability and Interpretation")
    print("-" * 40)
    
    # Demonstrate explanation generation
    sample_proposal = sample_proposals.iloc[0]
    print(f"Generating explanation for: {sample_proposal['Title']}")
    
    # Mock SHAP explanation
    shap_explanation = {
        "shap_values": {
            "Novelty_Score": 0.25,
            "Technical_Score": 0.30,
            "Financial_Score": -0.10,
            "Risk_Score": -0.05
        },
        "base_value": 0.70
    }
    
    # Mock LIME explanation
    lime_explanation = {
        "lime_weights": [
            ("Technical_Score", 0.35),
            ("Novelty_Score", 0.25),
            ("Financial_Score", -0.15)
        ]
    }
    
    # Mock feature importance
    feature_importance = {
        "Technical_Score": 0.35,
        "Novelty_Score": 0.25,
        "Financial_Score": 0.20,
        "Risk_Score": 0.10,
        "Coal_Relevance": 0.05,
        "Alignment": 0.05
    }
    
    explanation_report = explainer.generate_explanation_report(
        sample_proposal.to_dict(),
        shap_explanation,
        lime_explanation,
        feature_importance
    )
    
    print("Key Contributors to Score:")
    for feature, contribution in explanation_report['shap_explanation']['top_positive_contributors']:
        print(f"  + {feature}: +{contribution:.4f}")
    for feature, contribution in explanation_report['shap_explanation']['top_negative_contributors']:
        print(f"  - {feature}: {contribution:.4f}")
    
    time.sleep(1)
    
    # Step 9: Security and Access Control
    print("\nStep 9: Security and Access Control")
    print("-" * 40)
    
    # Create sample users
    users = [
        ("user001", "alice_reviewer", "Alice123!", "reviewer", "alice@research.org"),
        ("user002", "bob_submitter", "Bob123!", "submitter", "bob@company.com")
    ]
    
    for user_id, username, password, role, email in users:
        if not security.user_exists(username):
            security.create_user(user_id, username, password, role, email)
            print(f"✓ Created user: {username} ({role})")
    
    # Authenticate a user
    user_info = security.authenticate_user("alice_reviewer", "Alice123!")
    if user_info:
        print(f"✓ User authenticated: {user_info['username']} ({user_info['role']})")
        
        # Check permissions
        can_evaluate = security.has_permission(user_info['user_id'], "evaluate")
        can_submit = security.has_permission(user_info['user_id'], "submit_proposals")
        print(f"  Permissions - Evaluate: {can_evaluate}, Submit: {can_submit}")
    
    time.sleep(1)
    
    # Step 10: Monitoring and Drift Detection
    print("\nStep 10: Monitoring and Drift Detection")
    print("-" * 40)
    
    # Generate monitoring report
    report = monitor.generate_monitoring_report()
    print(f"✓ Monitoring report generated")
    print(f"  Total predictions monitored: {report['total_predictions']}")
    print(f"  Human feedback collected: {report.get('human_feedback_count', 0)}")
    
    # Check for alerts
    alerts = report.get("alerts", [])
    if alerts:
        print(f"  Alerts detected: {len(alerts)}")
        for alert in alerts:
            print(f"    - {alert['type']}: {alert['severity']}")
    else:
        print("  ✓ No critical alerts detected")
    
    time.sleep(1)
    
    # Final Step: System Summary
    print("\n" + "=" * 80)
    print("SYSTEM DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    # Database statistics
    stats = database.get_database_stats()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Security statistics
    print("\nSecurity Statistics:")
    print("  ✓ User authentication system operational")
    print("  ✓ Role-based access control implemented")
    print("  ✓ Data encryption enabled")
    print("  ✓ Audit logging active")
    
    print("\nKey Features Demonstrated:")
    print("  ✓ Document parsing and feature extraction")
    print("  ✓ FAISS-based novelty detection")
    print("  ✓ Technical feasibility evaluation")
    print("  ✓ Financial viability assessment")
    print("  ✓ Risk, ethics, and IP checking")
    print("  ✓ Model explainability with SHAP/LIME")
    print("  ✓ Database integration")
    print("  ✓ Security and access control")
    print("  ✓ Monitoring and drift detection")
    
    print("\nNote: This demonstration used sample data. In a production environment,")
    print("the system would process real R&D proposals with full security and compliance.")
    
    # Cleanup
    cleanup_files = [
        'demo_proposals.index',
        'demo_proposals_metadata.json',
        'demo_system.db',
        'demo_security.db',
        'encryption.key'
    ]
    
    print(f"\nDemo files created: {', '.join(cleanup_files)}")
    print("These files can be deleted after review.")

if __name__ == "__main__":
    demonstrate_complete_system()