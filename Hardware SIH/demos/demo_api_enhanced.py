import requests
import json
import time
import threading
from api_enhanced import app
import pandas as pd
from enhanced_evaluator import create_sample_data

def demo_api_endpoints():
    """Demonstrate the enhanced API endpoints"""
    print("Demonstrating Enhanced R&D Proposal Evaluation API")
    print("=" * 55)
    
    # Create sample data
    create_sample_data()
    print("✓ Sample data created")
    
    # Start the Flask app in a separate thread
    def run_app():
        app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)
    
    # Start server in background
    server_thread = threading.Thread(target=run_app)
    server_thread.daemon = True
    server_thread.start()
    
    # Wait a moment for server to start
    time.sleep(2)
    
    base_url = "http://127.0.0.1:5000"
    
    try:
        # 1. Health check
        print("\n1. Health Check:")
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        # 2. Get current weights
        print("\n2. Get Current Weights:")
        response = requests.get(f"{base_url}/weights")
        if response.status_code == 200:
            weights = response.json()
            print(f"   Status: {response.status_code}")
            print(f"   Weights: {weights['weights']}")
        else:
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.json()}")
        
        # 3. Update weights
        print("\n3. Update Weights:")
        new_weights = {
            "weights": {
                "novelty": 0.30,
                "financial": 0.10,
                "technical": 0.20,
                "coal_relevance": 0.10,
                "alignment": 0.10,
                "clarity": 0.10,
                "impact": 0.10
            }
        }
        response = requests.post(f"{base_url}/weights", json=new_weights)
        if response.status_code == 200:
            result = response.json()
            print(f"   Status: {response.status_code}")
            print(f"   Updated Weights: {result['weights']}")
        else:
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.json()}")
        
        # 4. Evaluate proposals
        print("\n4. Evaluate Proposals:")
        evaluation_data = {
            "past_proposals_path": "sample_past_proposals.csv",
            "new_proposals_path": "sample_new_proposals.csv"
        }
        response = requests.post(f"{base_url}/proposals/evaluate", json=evaluation_data)
        if response.status_code == 200:
            result = response.json()
            print(f"   Status: {response.status_code}")
            print(f"   Number of proposals evaluated: {len(result['results'])}")
            if result['results']:
                first_proposal = result['results'][0]
                print(f"   First proposal: {first_proposal['Title']}")
                print(f"   Overall Score: {first_proposal['Overall_Score']}")
        else:
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.json()}")
        
        # 5. Get proposal report
        print("\n5. Get Proposal Report:")
        response = requests.get(f"{base_url}/proposals/N001/report")
        if response.status_code == 200:
            result = response.json()
            print(f"   Status: {response.status_code}")
            print(f"   Proposal ID: {result['report']['proposal_id']}")
            print(f"   Title: {result['report']['title']}")
        else:
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.json()}")
        
        # 6. Submit feedback
        print("\n6. Submit Feedback:")
        feedback_data = {
            "reviewer_id": "DEMO_REVIEWER",
            "feedback": {
                "comments": "Excellent proposal with strong technical approach",
                "rating": 4.8,
                "accept": True
            }
        }
        response = requests.post(f"{base_url}/proposals/N001/feedback", json=feedback_data)
        if response.status_code == 200:
            result = response.json()
            print(f"   Status: {response.status_code}")
            print(f"   Message: {result['message']}")
        else:
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.json()}")
        
        # 7. Search similar proposals
        print("\n7. Search Similar Proposals:")
        response = requests.get(f"{base_url}/search/similar?proposal_id=N001&k=3")
        if response.status_code == 200:
            result = response.json()
            print(f"   Status: {response.status_code}")
            print(f"   Number of similar proposals: {len(result['similar_proposals'])}")
        else:
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.json()}")
        
        # 8. Get monitoring report
        print("\n8. Get Monitoring Report:")
        response = requests.get(f"{base_url}/monitoring/report")
        if response.status_code == 200:
            result = response.json()
            print(f"   Status: {response.status_code}")
            report = result['monitoring_report']
            print(f"   Total predictions: {report['total_predictions']}")
        else:
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.json()}")
        
        # 9. Get monitoring alerts
        print("\n9. Get Monitoring Alerts:")
        response = requests.get(f"{base_url}/monitoring/alerts")
        if response.status_code == 200:
            result = response.json()
            print(f"   Status: {response.status_code}")
            print(f"   Retraining needed: {result['retraining_needed']}")
            print(f"   Number of alerts: {len(result['alerts'])}")
        else:
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.json()}")
        
        # 10. Get explanation
        print("\n10. Get Evaluation Explanation:")
        response = requests.get(f"{base_url}/explain/N001")
        if response.status_code == 200:
            result = response.json()
            print(f"   Status: {response.status_code}")
            explanation = result['explanation']
            print(f"   Proposal ID: {explanation['proposal_id']}")
            print(f"   SHAP features: {len(explanation['shap_explanation']['top_positive_contributors'])}")
        else:
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.json()}")
        
        print("\n" + "=" * 55)
        print("✓ Enhanced API demonstration completed successfully")
        print("\nNote: The API server is still running in the background.")
        print("Press Ctrl+C to stop the server.")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server. Make sure it's running.")
    except Exception as e:
        print(f"Error during demonstration: {e}")

if __name__ == "__main__":
    demo_api_endpoints()