from flask import Flask, request, jsonify, send_file
import pandas as pd
import os
import json
import uuid
from datetime import datetime
from functools import wraps
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.enhanced_evaluator_hil import EnhancedRDPEvaluatorHIL
from utils.document_parser import DocumentParser
from utils.pdf_generator import PDFGenerator
from models.novelty_detector import FAISSNoveltyDetector
from models.technical_feasibility import TechnicalFeasibilityEvaluator
from models.financial_viability import FinancialViabilityEvaluator
from models.risk_ethics_ip import RiskEthicsIPChecker
from models.explainability import ModelExplainer
from models.model_monitoring import ModelMonitor
from database.database import ProposalDatabase
from utils.security import SecurityManager

app = Flask(__name__)

# Initialize components
evaluator = EnhancedRDPEvaluatorHIL()
monitor = ModelMonitor("proposal_evaluator")
database = ProposalDatabase("proposals.db")
security = SecurityManager("security.db")

# Ensure upload directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def require_auth(f):
    """Decorator to require authentication for API endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get session token from header
        session_token = request.headers.get('Authorization')
        if not session_token:
            return jsonify({
                "status": "error",
                "message": "Authorization token required"
            }), 401
        
        # Validate session
        session_info = security.validate_session(session_token)
        if not session_info:
            return jsonify({
                "status": "error",
                "message": "Invalid or expired session"
            }), 401
        
        # Add user info to request context
        setattr(request, 'user_info', session_info)
        
        # Log the API access
        security.log_audit_event(
            user_id=session_info['user_id'],
            action=f"api_access_{request.endpoint}",
            resource="api",
            details={"endpoint": request.endpoint, "method": request.method},
            ip_address=request.remote_addr or 'unknown',
            success=True
        )
        
        return f(*args, **kwargs)
    return decorated_function

def require_permission(permission):
    """Decorator to require specific permission for API endpoints"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # This assumes require_auth has already been applied
            user_id = getattr(request, 'user_info', {})['user_id']
            
            if not security.has_permission(user_id, permission):
                security.log_audit_event(
                    user_id=user_id,
                    action=f"permission_denied_{request.endpoint}",
                    resource="api",
                    details={"endpoint": request.endpoint, "required_permission": permission},
                    ip_address=request.remote_addr or 'unknown',
                    success=False
                )
                
                return jsonify({
                    "status": "error",
                    "message": f"Permission denied: {permission} required"
                }), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Secure R&D Proposal Evaluation API is running"})

@app.route('/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        if request.json is None:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided"
            }), 400
        
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({
                "status": "error",
                "message": "Username and password required"
            }), 400
        
        # Authenticate user
        user_info = security.authenticate_user(username, password)
        if not user_info:
            security.log_audit_event(
                user_id=None,
                action="login_failed",
                resource="auth",
                details={"username": username},
                ip_address=request.remote_addr or 'unknown',
                success=False
            )
            
            return jsonify({
                "status": "error",
                "message": "Invalid username or password"
            }), 401
        
        # Create session
        session_id = security.create_session(
            user_info['user_id'],
            ip_address=request.remote_addr or 'unknown',
            user_agent=request.headers.get('User-Agent', '')
        )
        
        security.log_audit_event(
            user_id=user_info['user_id'],
            action="login_success",
            resource="auth",
            details={"username": username},
            ip_address=request.remote_addr or 'unknown',
            success=True
        )
        
        return jsonify({
            "status": "success",
            "message": "Login successful",
            "session_token": session_id,
            "user_info": {
                "user_id": user_info['user_id'],
                "username": user_info['username'],
                "role": user_info['role'],
                "email": user_info['email']
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/auth/logout', methods=['POST'])
@require_auth
def logout():
    """User logout endpoint"""
    try:
        session_token = request.headers.get('Authorization', '')
        
        # Invalidate session
        security.invalidate_session(session_token)
        
        security.log_audit_event(
            user_id=getattr(request, 'user_info', {}).get('user_id', 'unknown'),
            action="logout",
            resource="auth",
            details={},
            ip_address=request.remote_addr or 'unknown',
            success=True
        )
        
        return jsonify({
            "status": "success",
            "message": "Logout successful"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/proposals/upload', methods=['POST'])
@require_auth
@require_permission('submit_proposals')
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
        if file.filename and file.filename.lower().endswith('.pdf'):
            sections = parser.parse_pdf(file_path)
        elif file.filename and file.filename.lower().endswith('.docx'):
            sections = parser.parse_docx(file_path)
        else:
            return jsonify({
                "status": "error",
                "message": "Unsupported file format. Please upload PDF or DOCX files."
            }), 400
        
        # Convert to evaluation format
        proposal_id = f"PROP_{uuid.uuid4().hex[:8].upper()}"
        csv_data = parser.convert_to_csv_format(sections, proposal_id)
        
        # Save to database
        proposal_data = {
            'proposal_id': proposal_id,
            'title': csv_data.get('Title', ''),
            'abstract': csv_data.get('Abstract', ''),
            'funding_requested': csv_data.get('Funding_Requested', 0.0),
            'pi_name': getattr(request, 'user_info', {}).get('username', ''),  # Set submitter as PI for demo
            'organization': 'Submitted via API',
            'contact_email': getattr(request, 'user_info', {}).get('email', ''),
            'sections': sections
        }
        
        database.insert_proposal(proposal_data)
        
        security.log_audit_event(
            user_id=getattr(request, 'user_info', {}).get('user_id', 'unknown') if hasattr(request, 'user_info') else 'unknown',
            action="proposal_upload",
            resource="proposals",
            details={"proposal_id": proposal_id, "filename": file.filename},
            ip_address=request.remote_addr or 'unknown',
            success=True
        )
        
        return jsonify({
            "status": "success",
            "message": "Proposal uploaded and parsed successfully",
            "proposal_id": proposal_id,
            "parsed_data": csv_data,
            "file_path": file_path
        })
        
    except Exception as e:
        security.log_audit_event(
            user_id=getattr(request, 'user_info', {}).get('user_id', 'unknown'),
            action="proposal_upload_failed",
            resource="proposals",
            details={"error": str(e)},
            ip_address=request.remote_addr or 'unknown',
            success=False
        )
        
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/proposals/evaluate', methods=['POST'])
@require_auth
@require_permission('evaluate')
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
        
        # Log predictions for monitoring and save to database
        for result in results_json:
            # Log for monitoring
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
            
            # Save evaluation scores to database
            scores_data = {
                'novelty_score': result['Novelty_Score'],
                'financial_score': result['Financial_Score'],
                'technical_score': result['Technical_Score'],
                'coal_relevance_score': result['Coal_Relevance_Score'],
                'alignment_score': result['Alignment_Score'],
                'clarity_score': result['Clarity_Score'],
                'impact_score': result['Impact_Score'],
                'overall_score': result['Overall_Score']
            }
            
            database.insert_evaluation_scores(result['Proposal_ID'], scores_data)
        
        security.log_audit_event(
            user_id=getattr(request, 'user_info', {}).get('user_id', 'unknown'),
            action="proposals_evaluate",
            resource="evaluations",
            details={"proposals_count": len(results_json)},
            ip_address=request.remote_addr or 'unknown',
            success=True
        )
        
        return jsonify({
            "status": "success",
            "results": results_json,
            "weights_used": evaluator.get_weights()
        })
    except Exception as e:
        security.log_audit_event(
            user_id=getattr(request, 'user_info', {}).get('user_id', 'unknown') if hasattr(request, 'user_info') else 'unknown',
            action="proposals_evaluate_failed",
            resource="evaluations",
            details={"error": str(e)},
            ip_address=request.remote_addr or 'unknown',
            success=False
        )
        
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/proposals/<proposal_id>/report', methods=['GET'])
@require_auth
@require_permission('read_proposals')
def get_proposal_report(proposal_id):
    """Get detailed evaluation report for a specific proposal"""
    try:
        # Check if user has access to this specific proposal
        # In a real implementation, you would check access control rules
        # For now, we'll allow access to all proposals for users with read permission
        
        # Get proposal from database
        proposal_data = database.get_proposal(proposal_id)
        if not proposal_data:
            return jsonify({
                "status": "error",
                "message": f"Proposal with ID {proposal_id} not found"
            }), 404
        
        # Get evaluation scores
        evaluation_scores = database.get_evaluation_scores(proposal_id)
        
        # Get feedback
        feedback = database.get_feedback(proposal_id)
        
        # Create comprehensive report
        report = {
            "proposal_id": proposal_id,
            "proposal_data": proposal_data,
            "evaluation_scores": evaluation_scores,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        
        security.log_audit_event(
            user_id=getattr(request, 'user_info', {}).get('user_id', 'unknown'),
            action="proposal_report_access",
            resource="proposals",
            details={"proposal_id": proposal_id},
            ip_address=request.remote_addr or 'unknown',
            success=True
        )
        
        return jsonify({
            "status": "success",
            "report": report
        })
        
    except Exception as e:
        security.log_audit_event(
            user_id=getattr(request, 'user_info', {}).get('user_id', 'unknown') if hasattr(request, 'user_info') else 'unknown',
            action="proposal_report_access_failed",
            resource="proposals",
            details={"proposal_id": proposal_id, "error": str(e)},
            ip_address=request.remote_addr or 'unknown',
            success=False
        )
        
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/proposals/<proposal_id>/feedback', methods=['POST'])
@require_auth
@require_permission('feedback')
def submit_feedback(proposal_id):
    """Submit human feedback on a proposal evaluation"""
    try:
        if request.json is None:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided"
            }), 400
        
        feedback_data = request.json
        reviewer_id = getattr(request, 'user_info', {}).get('user_id', 'unknown') if hasattr(request, 'user_info') else 'unknown'
        
        # Save feedback to database
        feedback_id = database.insert_feedback(proposal_id, reviewer_id, feedback_data)
        
        # Log feedback for monitoring
        monitor.log_prediction(
            proposal_id=proposal_id,
            features={},  # In a real implementation, we'd include actual features
            prediction=0.0,  # Placeholder
            human_feedback=feedback_data
        )
        
        security.log_audit_event(
            user_id=getattr(request, 'user_info', {}).get('user_id', 'unknown'),
            action="feedback_submit",
            resource="feedback",
            details={"proposal_id": proposal_id, "feedback_id": feedback_id},
            ip_address=request.remote_addr or 'unknown',
            success=True
        )
        
        return jsonify({
            "status": "success",
            "message": "Feedback submitted successfully",
            "feedback_id": feedback_id
        })
        
    except Exception as e:
        security.log_audit_event(
            user_id=getattr(request, 'user_info', {}).get('user_id', 'unknown') if hasattr(request, 'user_info') else 'unknown',
            action="feedback_submit_failed",
            resource="feedback",
            details={"proposal_id": proposal_id, "error": str(e)},
            ip_address=request.remote_addr or 'unknown',
            success=False
        )
        
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/monitoring/report', methods=['GET'])
@require_auth
@require_permission('all')
def get_monitoring_report():
    """Get model monitoring report"""
    try:
        report = monitor.generate_monitoring_report()
        
        security.log_audit_event(
            user_id=getattr(request, 'user_info', {}).get('user_id', 'unknown'),
            action="monitoring_report_access",
            resource="monitoring",
            details={},
            ip_address=request.remote_addr or 'unknown',
            success=True
        )
        
        return jsonify({
            "status": "success",
            "monitoring_report": report
        })
        
    except Exception as e:
        security.log_audit_event(
            user_id=getattr(request, 'user_info', {}).get('user_id', 'unknown'),
            action="monitoring_report_access_failed",
            resource="monitoring",
            details={"error": str(e)},
            ip_address=request.remote_addr or 'unknown',
            success=False
        )
        
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/audit/log', methods=['GET'])
@require_auth
@require_permission('all')
def get_audit_log():
    """Get audit log"""
    try:
        # Get query parameters
        user_id = request.args.get('user_id')
        action = request.args.get('action')
        limit = int(request.args.get('limit', 100))
        
        # Get audit log from security manager
        audit_logs = security.get_audit_log(user_id=user_id, action=action, limit=limit)
        
        security.log_audit_event(
            user_id=getattr(request, 'user_info', {}).get('user_id', 'unknown'),
            action="audit_log_access",
            resource="audit",
            details={"filters": {"user_id": user_id, "action": action, "limit": limit}},
            ip_address=request.remote_addr or 'unknown',
            success=True
        )
        
        return jsonify({
            "status": "success",
            "audit_logs": audit_logs
        })
        
    except Exception as e:
        security.log_audit_event(
            user_id=getattr(request, 'user_info', {}).get('user_id', 'unknown'),
            action="audit_log_access_failed",
            resource="audit",
            details={"error": str(e)},
            ip_address=request.remote_addr or 'unknown',
            success=False
        )
        
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    print("Starting Secure R&D Proposal Evaluation API...")
    print("API endpoints available at http://localhost:5000")
    print("\nTo use the API:")
    print("1. POST /auth/login - Login with username and password")
    print("2. Use the returned session_token in the Authorization header for subsequent requests")
    print("3. All other endpoints require authentication and appropriate permissions")
    
    app.run(debug=True, host='0.0.0.0', port=5000)