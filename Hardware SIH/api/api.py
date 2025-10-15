from flask import Flask, request, jsonify, send_file
import pandas as pd
import os
import sys
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.enhanced_evaluator import EnhancedRDPEvaluator, create_sample_data

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "R&D Proposal Evaluation API is running"})

@app.route('/evaluate', methods=['POST'])
def evaluate_proposals():
    """Evaluate proposals endpoint"""
    try:
        # Get request data
        if request.json is None:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided"
            }), 400
            
        data = request.json
        past_proposals_path = data.get('past_proposals_path', 'sample_past_proposals.csv')
        new_proposals_path = data.get('new_proposals_path', 'sample_new_proposals.csv')
        include_feasibility = data.get('include_feasibility', False)
        model_type = data.get('model_type', 'sentence_bert')
        
        # Create evaluator with specified model type
        evaluator = EnhancedRDPEvaluator(model_type=model_type)
        
        # Evaluate proposals
        results = evaluator.evaluate_proposals(
            past_proposals_path, 
            new_proposals_path
        )
        
        # Convert to JSON format
        results_json = results.to_dict(orient='records')
        
        return jsonify({
            "status": "success",
            "results": results_json
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/sample-data', methods=['POST'])
def generate_sample_data():
    """Generate sample data"""
    try:
        create_sample_data()
        return jsonify({
            "status": "success",
            "message": "Sample data created successfully"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/download-results', methods=['POST'])
def download_results():
    """Download evaluation results as CSV"""
    try:
        # Get request data
        if request.json is None:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided"
            }), 400
            
        data = request.json
        past_proposals_path = data.get('past_proposals_path', 'sample_past_proposals.csv')
        new_proposals_path = data.get('new_proposals_path', 'sample_new_proposals.csv')
        output_path = data.get('output_path', 'evaluated_proposals.csv')
        
        # Create evaluator
        evaluator = EnhancedRDPEvaluator()
        
        # Evaluate proposals
        results = evaluator.evaluate_proposals(
            past_proposals_path, 
            new_proposals_path
        )
        
        # Save results
        evaluator.save_results(results, output_path)
        
        # Return file
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    print("Starting R&D Proposal Evaluation API...")
    app.run(debug=True, host='0.0.0.0', port=5000)