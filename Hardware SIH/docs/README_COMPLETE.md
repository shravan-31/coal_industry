# Advanced Auto Evaluation System for R&D Proposals

## Overview

This repository contains a complete implementation of an Advanced Auto Evaluation System for R&D Proposals as specified in the design requirements. The system provides automated evaluation of research proposals using AI/ML techniques with human-in-the-loop capabilities, explainability features, and comprehensive security.

## System Architecture

The system is composed of the following key modules:

1. **Document Ingestion & Parsing**
2. **Knowledge Base & Vector Index**
3. **Novelty Detection**
4. **Technical Feasibility Module**
5. **Financial Viability Module**
6. **Risk & Ethics Checks**
7. **Final Scoring & Calibration**
8. **Explainability & Audit**
9. **Human-in-the-loop UI**
10. **Security & Compliance**
11. **Monitoring & Retraining**

## Key Features

### Evaluation Modules
- **Semantic Novelty Detection** using FAISS vector database
- **Technical Feasibility Assessment** with BERT and ensemble methods
- **Financial Viability Analysis** with rule engine and anomaly detection
- **Risk, Ethics & IP Checking** with comprehensive screening
- **Multi-criteria Scoring** with customizable weights

### Explainability
- **SHAP Integration** for model interpretability
- **LIME Implementation** for local explanations
- **Feature Importance Analysis** for tree-based models
- **Text Highlighting** for influential content

### Human-in-the-loop
- **Dynamic Weight Adjustment** for evaluation parameters
- **Interactive Feedback Collection** from reviewers
- **Real-time Re-scoring** with adjusted weights
- **Override Mechanisms** for human judgment

### Security & Compliance
- **Role-based Access Control** (RBAC)
- **Data Encryption** for sensitive information
- **Audit Logging** for compliance tracking
- **Session Management** with expiration

### Monitoring
- **Feature Drift Detection** with statistical tests
- **Prediction Drift Monitoring** with alerts
- **Performance Evaluation** with standard metrics
- **Human-Model Agreement** tracking

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation Steps

1. **Clone the repository:**
```bash
git clone <repository-url>
cd hardware-sih
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

3. **Install additional packages for advanced features:**
```bash
pip install shap lime faiss-cpu cryptography
```

### Required Packages
- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=1.0.0
- sentence-transformers>=2.2.0
- nltk>=3.6.0
- streamlit>=1.10.0
- xgboost>=1.7.0
- lightgbm>=3.3.0
- transformers>=4.20.0
- torch>=1.12.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- plotly>=5.10.0
- flask>=2.2.0
- PyPDF2>=3.0.0
- python-docx>=0.8.11
- reportlab>=3.6.0
- shap>=0.42.0
- lime>=0.2.0
- faiss-cpu>=1.7.0
- cryptography>=3.4.0

## Usage

### 1. Web Interface (Streamlit)
To run the web interface:
```bash
streamlit run app_hil.py
```

The interface will be available at `http://localhost:8501`

### 2. REST API
To run the REST API:
```bash
python api_secure.py
```

The API will be available at `http://localhost:5000`

### 3. Command Line Tools
Each module can be run independently:

```bash
# Novelty detection
python demo_novelty_detector.py

# Technical feasibility
python demo_technical_feasibility.py

# Financial viability
python demo_financial_viability.py

# Risk, ethics, IP checks
python demo_risk_ethics_ip.py

# Explainability
python demo_explainability.py

# Model monitoring
python demo_model_monitoring.py

# Database operations
python demo_database.py

# Security features
python demo_security.py

# Complete system demonstration
python demo_complete_system.py
```

## System Components

### Document Parser (`document_parser.py`)
- Parses PDF and DOCX documents
- Extracts key sections (abstract, objectives, methodology, etc.)
- Converts documents to evaluation format

### Novelty Detector (`novelty_detector.py`)
- FAISS-based vector database for semantic similarity
- Sentence-BERT embeddings for proposal representation
- Novelty score calculation and similar proposal search

### Technical Feasibility Evaluator (`technical_feasibility.py`)
- BERT classifier for feasibility prediction
- XGBoost/LightGBM ensemble models
- Component scoring (team, budget, timeline, methodology)

### Financial Viability Evaluator (`financial_viability.py`)
- Rule engine for funding guideline compliance
- Anomaly detection for unusual budgets
- Budget z-score calculation and scoring

### Risk, Ethics & IP Checker (`risk_ethics_ip.py`)
- Environmental and safety risk assessment
- Ethics compliance checking
- Conflict of interest detection
- IP conflict identification

### Explainability Module (`explainability.py`)
- SHAP integration for model interpretability
- LIME implementation for local explanations
- Feature importance analysis
- Text span highlighting

### Enhanced Evaluator (`enhanced_evaluator_hil.py`)
- Main evaluation engine with human-in-the-loop features
- Dynamic weight adjustment
- Feedback collection and processing

### Model Monitor (`model_monitoring.py`)
- Feature and prediction drift detection
- Model performance evaluation
- Human-model agreement tracking
- Retraining alert system

### Database (`database.py`)
- SQLite-based relational database
- Proposal, evaluation, feedback storage
- Audit logging and export capabilities

### Security (`security.py`)
- User authentication and session management
- Role-based access control
- Data encryption
- Audit logging

### APIs (`api_enhanced.py`, `api_secure.py`)
- RESTful endpoints for all system functionalities
- Secure authentication with RBAC
- Comprehensive API documentation

## Data Schema

### Proposals Table
- `proposal_id`: Unique identifier
- `title`: Proposal title
- `abstract`: Proposal abstract
- `funding_requested`: Requested funding amount
- `submission_date`: Submission timestamp
- `status`: Current status
- Additional metadata fields

### Evaluation Scores Table
- `score_id`: Unique identifier
- `proposal_id`: Reference to proposal
- Component scores (novelty, technical, financial, etc.)
- `overall_score`: Final evaluation score
- `evaluation_date`: Timestamp

### Feedback Table
- `feedback_id`: Unique identifier
- `proposal_id`: Reference to proposal
- `reviewer_id`: Reviewer identifier
- Feedback content and ratings
- `feedback_date`: Timestamp

## Security Features

### Authentication
- Password hashing with salt
- Session management with expiration
- Secure token-based authentication

### Authorization
- Role-based access control (RBAC)
- Permission hierarchies
- Resource-level access control

### Data Protection
- Encryption for sensitive data
- Secure communication (HTTPS recommended)
- Audit trails for compliance

## Monitoring and Maintenance

### Drift Detection
- Statistical tests for feature distribution changes
- Population Stability Index (PSI) monitoring
- Wasserstein distance for distribution comparison

### Performance Monitoring
- Accuracy, precision, recall, F1 metrics
- Human-model agreement tracking
- Performance degradation alerts

### Retraining System
- Automated retraining triggers
- Feedback-based weight adjustment
- Model version management

## Testing

Each module includes comprehensive unit tests:
```bash
# Run tests for individual modules
python test_novelty_detector.py
python test_technical_feasibility.py
python test_financial_viability.py
python test_risk_ethics_ip.py
python test_explainability.py
python test_enhanced_evaluator_hil.py
python test_model_monitoring.py
python test_database.py
python test_security.py
```

## Deployment

### Local Development
1. Install dependencies
2. Run Streamlit UI or Flask API
3. Access through browser or API clients

### Production Deployment
1. Use Docker containers for isolation
2. Deploy with Kubernetes or similar orchestration
3. Configure reverse proxy (nginx) for HTTPS
4. Set up monitoring and logging systems
5. Implement backup and disaster recovery

### Environment Variables
- `FLASK_ENV`: Development/production mode
- `SECRET_KEY`: Application secret for security
- `DATABASE_URL`: Database connection string
- `ENCRYPTION_KEY`: Key for data encryption

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Implementation team for NaCCER / CMPDI R&D Proposal Evaluation System

## Acknowledgments

- Sentence-BERT for semantic embeddings
- FAISS for vector similarity search
- SHAP and LIME for model explainability
- Open-source ML libraries and frameworks

## Support

For issues and questions, please create an issue in the repository or contact the development team.