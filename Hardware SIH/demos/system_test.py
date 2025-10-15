"""
System Test Script for R&D Proposal Evaluation System
"""
import requests
import time
import threading
from api_secure import app
import json

def start_api():
    """Start the Flask API in a separate thread"""
    app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)

def test_system():
    """Test the complete system"""
    print("Testing R&D Proposal Evaluation System")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing Health Check Endpoint:")
    try:
        response = requests.get('http://127.0.0.1:5000/health')
        if response.status_code == 200:
            print("   ✓ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"   ✗ Health check failed with status {response.status_code}")
    except Exception as e:
        print(f"   ✗ Health check failed: {e}")
    
    # Test 2: Sample data generation
    print("\n2. Testing Sample Data Generation:")
    try:
        from enhanced_evaluator import create_sample_data
        create_sample_data()
        print("   ✓ Sample data created")
    except Exception as e:
        print(f"   ✗ Sample data creation failed: {e}")
    
    # Test 3: Novelty detection
    print("\n3. Testing Novelty Detection:")
    try:
        from novelty_detector import create_proposal_kb_from_csv
        create_proposal_kb_from_csv(
            'sample_past_proposals.csv', 
            'test_proposals.index', 
            'test_proposals_metadata.json'
        )
        print("   ✓ Knowledge base created")
    except Exception as e:
        print(f"   ✗ Knowledge base creation failed: {e}")
    
    # Test 4: Technical feasibility
    print("\n4. Testing Technical Feasibility Module:")
    try:
        from technical_feasibility import TechnicalFeasibilityEvaluator
        evaluator = TechnicalFeasibilityEvaluator()
        print("   ✓ Technical feasibility evaluator initialized")
    except Exception as e:
        print(f"   ✗ Technical feasibility evaluator failed: {e}")
    
    # Test 5: Financial viability
    print("\n5. Testing Financial Viability Module:")
    try:
        from financial_viability import FinancialViabilityEvaluator
        evaluator = FinancialViabilityEvaluator()
        print("   ✓ Financial viability evaluator initialized")
    except Exception as e:
        print(f"   ✗ Financial viability evaluator failed: {e}")
    
    # Test 6: Risk, ethics, IP checks
    print("\n6. Testing Risk, Ethics & IP Module:")
    try:
        from risk_ethics_ip import RiskEthicsIPChecker
        checker = RiskEthicsIPChecker()
        print("   ✓ Risk, ethics & IP checker initialized")
    except Exception as e:
        print(f"   ✗ Risk, ethics & IP checker failed: {e}")
    
    # Test 7: Explainability
    print("\n7. Testing Explainability Module:")
    try:
        from explainability import ModelExplainer
        explainer = ModelExplainer()
        print("   ✓ Model explainer initialized")
    except Exception as e:
        print(f"   ✗ Model explainer failed: {e}")
    
    # Test 8: Database
    print("\n8. Testing Database Module:")
    try:
        from database import ProposalDatabase
        db = ProposalDatabase("test_system.db")
        print("   ✓ Database initialized")
    except Exception as e:
        print(f"   ✗ Database initialization failed: {e}")
    
    # Test 9: Security
    print("\n9. Testing Security Module:")
    try:
        from security import SecurityManager
        security = SecurityManager("test_security.db")
        print("   ✓ Security manager initialized")
    except Exception as e:
        print(f"   ✗ Security manager failed: {e}")
    
    # Test 10: Model monitoring
    print("\n10. Testing Model Monitoring Module:")
    try:
        from model_monitoring import ModelMonitor
        monitor = ModelMonitor("test_model")
        print("   ✓ Model monitor initialized")
    except Exception as e:
        print(f"   ✗ Model monitor failed: {e}")
    
    print("\n" + "=" * 50)
    print("System test completed!")
    print("\nTo interact with the full system:")
    print("1. Open your browser and go to http://localhost:8501 for the web interface")
    print("2. The API is running on http://localhost:5000")

if __name__ == "__main__":
    # Start API in background thread
    api_thread = threading.Thread(target=start_api)
    api_thread.daemon = True
    api_thread.start()
    
    # Wait a moment for API to start
    time.sleep(3)
    
    # Run tests
    test_system()