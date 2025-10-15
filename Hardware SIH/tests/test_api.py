import requests
import json

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:5000"
    
    # Test health endpoint
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check status code: {response.status_code}")
        print(f"Health check response: {response.json()}")
    except Exception as e:
        print(f"Error testing health endpoint: {e}")
    
    # Test sample data endpoint
    print("\nTesting sample data endpoint...")
    try:
        response = requests.post(f"{base_url}/sample-data")
        print(f"Sample data status code: {response.status_code}")
        print(f"Sample data response: {response.json()}")
    except Exception as e:
        print(f"Error testing sample data endpoint: {e}")
    
    # Test evaluation endpoint
    print("\nTesting evaluation endpoint...")
    try:
        payload = {
            "past_proposals_path": "sample_past_proposals.csv",
            "new_proposals_path": "sample_new_proposals.csv",
            "include_feasibility": True,
            "model_type": "sentence_bert"
        }
        response = requests.post(f"{base_url}/evaluate", json=payload)
        print(f"Evaluation status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Evaluation response status: {result['status']}")
            if result['status'] == 'success':
                print(f"Number of evaluated proposals: {len(result['results'])}")
                if len(result['results']) > 0:
                    print("First proposal evaluation:")
                    print(json.dumps(result['results'][0], indent=2))
            else:
                print(f"Error: {result.get('message', 'Unknown error')}")
        else:
            print(f"Error response: {response.text}")
    except Exception as e:
        print(f"Error testing evaluation endpoint: {e}")

if __name__ == "__main__":
    test_api()