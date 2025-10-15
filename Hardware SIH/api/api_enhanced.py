from flask import Flask, request, jsonify, send_file
import pandas as pd
import os
import json
import uuid
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.enhanced_evaluator_hil import EnhancedRDPEvaluatorHIL
from core.proposal_evaluator import create_sample_data
from utils.document_parser import DocumentParser
from utils.pdf_generator import PDFGenerator
from models.novelty_detector import FAISSNoveltyDetector, create_proposal_kb_from_csv
from models.technical_feasibility import TechnicalFeasibilityEvaluator
from models.financial_viability import FinancialViabilityEvaluator
from models.risk_ethics_ip import RiskEthicsIPChecker
from models.explainability import ModelExplainer
from models.model_monitoring import ModelMonitor

app = Flask(__name__)

# Initialize components
evaluator = EnhancedRDPEvaluatorHIL()
monitor = ModelMonitor("proposal_evaluator")

# Ensure upload directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "R&D Proposal Evaluation API is running"})

@app.route('/proposals/upload', methods=['POST'])
def upload_proposal():
    """Upload a proposal document (PDF/DOCX)"""
    try:
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No file provided"
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "status": "error",
                "message": "No file selected"
            }), 400
        
        # Generate unique filename
        filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save file
        file.save(file_path)
        
        # Parse document
        parser = DocumentParser()
        if file.filename and file.filename and file.filename.lower().endswith('.pdf'):
            sections = parser.parse_pdf(file_path)
        elif file.filename and file.filename and file.filename.lower().endswith('.docx'):
            sections = parser.parse_docx(file_path)
        else:
            return jsonify({
                "status": "error",
                "message": "Unsupported file format. Please upload PDF or DOCX files."
            }), 400
        
        # Convert to evaluation format
        proposal_id = f"PROP_{uuid.uuid4().hex[:8].upper()}"
        csv_data = parser.convert_to_csv_format(sections, proposal_id)
        
        return jsonify({
            "status": "success",
            "message": "Proposal uploaded and parsed successfully",
            "proposal_id": proposal_id,
            "parsed_data": csv_data,
            "file_path": file_path
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/proposals/evaluate', methods=['POST'])
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
        model_type = data.get('model_type', 'sentence_bert')
        custom_weights = data.get('weights', None)
        
        # Create evaluator with specified model type
        evaluator = EnhancedRDPEvaluatorHIL(model_type=model_type)
        
        # Set custom weights if provided
        if custom_weights:
            evaluator.set_weights(custom_weights)
        
        # Evaluate proposals
        results = evaluator.evaluate_proposals(past_proposals_path, new_proposals_path)
        
        # Convert to JSON format
        results_json = results.to_dict(orient='records')
        
        # Log predictions for monitoring
        for result in results_json:
            monitor.log_prediction(
                proposal_id=result['Proposal_ID'],
                features={
                    'novelty_score': result['Novelty_Score'],
                    'financial_score': result['Financial_Score'],
                    'technical_score': result['Technical_Score'],
                    'coal_relevance_score': result['Coal_Relevance_Score'],
                    'alignment_score': result['Alignment_Score'],
                    'clarity_score': result['Clarity_Score'],
                    'impact_score': result['Impact_Score']
                },
                prediction=result['Overall_Score'] / 100.0  # Normalize to 0-1
            )
        
        return jsonify({
            "status": "success",
            "results": results_json,
            "weights_used": evaluator.get_weights()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/proposals/<proposal_id>/report', methods=['GET'])
def get_proposal_report(proposal_id):
    """Get detailed evaluation report for a specific proposal"""
    try:
        # In a real implementation, this would fetch from a database
        # For now, we'll simulate with sample data
        if not os.path.exists('sample_new_proposals.csv'):
            create_sample_data()
        
        # Load sample data
        df = pd.read_csv('sample_new_proposals.csv')
        
        # Find proposal (in real implementation, this would be a database query)
        proposal = df[df['Proposal_ID'] == proposal_id]
        if proposal.empty:
            return jsonify({
                "status": "error",
                "message": f"Proposal with ID {proposal_id} not found"
            }), 404
        
        proposal_data = proposal.iloc[0].to_dict()
        
        # Generate detailed analysis
        technical_evaluator = TechnicalFeasibilityEvaluator()
        technical_result = technical_evaluator.evaluate_proposal(proposal_data)
        
        financial_evaluator = FinancialViabilityEvaluator()
        financial_result = financial_evaluator.evaluate_proposal(proposal_data)
        
        risk_checker = RiskEthicsIPChecker()
        risk_result = risk_checker.evaluate_proposal(proposal_data)
        
        # Create comprehensive report
        report = {
            "proposal_id": proposal_id,
            "title": proposal_data['Title'],
            "timestamp": datetime.now().isoformat(),
            "technical_feasibility": technical_result,
            "financial_viability": financial_result,
            "risk_ethics_ip": risk_result,
            "overall_assessment": "This is a simulated report. In a production environment, this would contain detailed analysis."
        }
        
        return jsonify({
            "status": "success",
            "report": report
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/proposals/<proposal_id>/feedback', methods=['POST'])
def submit_feedback(proposal_id):
    """Submit human feedback on a proposal evaluation"""
    try:
        if request.json is None:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided"
            }), 400
        
        feedback_data = request.json
        reviewer_id = feedback_data.get('reviewer_id', 'anonymous')
        feedback = feedback_data.get('feedback', {})
        override_scores = feedback_data.get('override_scores', None)
        
        # In a real implementation, this would save to a database
        # For now, we'll just log it and update our evaluator
        evaluator.collect_human_feedback(
            proposal_id=proposal_id,
            reviewer_id=reviewer_id,
            feedback=feedback,
            override_scores=override_scores
        )
        
        # Log feedback for monitoring
        monitor.log_prediction(
            proposal_id=proposal_id,
            features={},  # In a real implementation, we'd include actual features
            prediction=0.0,  # Placeholder
            human_feedback=feedback
        )
        
        return jsonify({
            "status": "success",
            "message": "Feedback submitted successfully"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/search/similar', methods=['GET'])
def search_similar_proposals():
    """Find similar proposals using FAISS"""
    try:
        proposal_id = request.args.get('proposal_id')
        k = int(request.args.get('k', 5))
        
        # In a real implementation, this would search in the FAISS index
        # For now, we'll simulate the response
        similar_proposals = [
            {
                "proposal_id": "SIM001",
                "title": "Similar Proposal 1",
                "similarity_score": 0.95,
                "verdict": "Approved"
            },
            {
                "proposal_id": "SIM002",
                "title": "Similar Proposal 2",
                "similarity_score": 0.87,
                "verdict": "Rejected"
            }
        ]
        
        return jsonify({
            "status": "success",
            "similar_proposals": similar_proposals
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/monitoring/report', methods=['GET'])
def get_monitoring_report():
    """Get model monitoring report"""
    try:
        report = monitor.generate_monitoring_report()
        
        return jsonify({
            "status": "success",
            "monitoring_report": report
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/monitoring/alerts', methods=['GET'])
def get_monitoring_alerts():
    """Get active monitoring alerts"""
    try:
        report = monitor.generate_monitoring_report()
        alerts = report.get("alerts", [])
        
        return jsonify({
            "status": "success",
            "alerts": alerts,
            "retraining_needed": monitor.trigger_retraining_alert()
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/weights', methods=['GET'])
def get_current_weights():
    """Get current evaluation weights"""
    try:
        weights = evaluator.get_weights()
        
        return jsonify({
            "status": "success",
            "weights": weights
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/weights', methods=['POST'])
def update_weights():
    """Update evaluation weights"""
    try:
        if request.json is None:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided"
            }), 400
        
        new_weights = request.json.get('weights', {})
        
        if evaluator.set_weights(new_weights):
            return jsonify({
                "status": "success",
                "message": "Weights updated successfully",
                "weights": evaluator.get_weights()
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Invalid weights - they must sum to 1.0"
            }), 400
        
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
        evaluator = EnhancedRDPEvaluatorHIL()
        
        # Evaluate proposals
        results = evaluator.evaluate_proposals(past_proposals_path, new_proposals_path)
        
        # Save results
        evaluator.save_results(results, output_path)
        
        # Return file
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/explain/<proposal_id>', methods=['GET'])
def explain_evaluation(proposal_id):
    """Get explanation for a proposal evaluation"""
    try:
        # In a real implementation, this would fetch the actual proposal and provide SHAP/LIME explanations
        # For now, we'll simulate the response
        
        explanation = {
            "proposal_id": proposal_id,
            "shap_explanation": {
                "top_positive_contributors": [
                    ["Technical_Score", 0.25],
                    ["Novelty_Score", 0.20],
                    ["Impact_Score", 0.15]
                ],
                "top_negative_contributors": [
                    ["Financial_Score", -0.10],
                    ["Clarity_Score", -0.05]
                ]
            },
            "lime_explanation": {
                "top_features": [
                    ["Technical_Score", 0.30],
                    ["Novelty_Score", 0.25],
                    ["Impact_Score", 0.20]
                ]
            },
            "feature_importance": {
                "Technical_Score": 0.30,
                "Novelty_Score": 0.25,
                "Impact_Score": 0.20,
                "Financial_Score": 0.10,
                "Coal_Relevance_Score": 0.05,
                "Alignment_Score": 0.05,
                "Clarity_Score": 0.05
            }
        }
        
        return jsonify({
            "status": "success",
            "explanation": explanation
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    print("Starting Enhanced R&D Proposal Evaluation API...")
    app.run(debug=True, host='0.0.0.0', port=5000)