# Project Structure

This document explains the organization of the R&D Proposal Evaluation System files.

## Directory Structure

```
Hardware SIH/
├── core/                 # Core evaluation logic and main applications
│   ├── enhanced_evaluator.py
│   ├── enhanced_evaluator_hil.py
│   ├── proposal_evaluator.py
│   ├── app.py
│   ├── app_hil.py
│   ├── startup.py
│   ├── create_sample_data.py
│   ├── sample_*.csv
│   └── uploaded_*.csv
├── models/               # Individual model components
│   ├── novelty_detector.py
│   ├── technical_feasibility.py
│   ├── financial_viability.py
│   ├── risk_ethics_ip.py
│   ├── explainability.py
│   └── model_monitoring.py
├── utils/                # Utility functions and helpers
│   ├── document_parser.py
│   ├── pdf_generator.py
│   └── security.py
├── api/                  # API endpoints and services
│   ├── api.py
│   ├── api_enhanced.py
│   ├── api_secure.py
│   └── simple_api.py
├── database/             # Database management
│   └── database.py
├── tests/                # Unit tests
│   ├── test_*.py
├── demos/                # Demonstration scripts
│   ├── demo_*.py
│   ├── quick_demo.py
│   ├── system_test.py
│   └── debug_test.py
├── docs/                 # Documentation
│   ├── README.md
│   ├── ENHANCED_README.md
│   ├── README_COMPLETE.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── SYSTEM_READY.md
│   ├── requirements.txt
│   ├── Dockerfile
│   └── docker-compose.yml
└── README_STRUCTURE.md   # This file
```

## Component Descriptions

### Core
Contains the main evaluation logic and application files:
- `enhanced_evaluator.py`: Main evaluation engine
- `enhanced_evaluator_hil.py`: Enhanced evaluator with human-in-the-loop features
- `proposal_evaluator.py`: Base proposal evaluator
- `app.py` and `app_hil.py`: Streamlit web applications
- `startup.py`: Application startup script
- `create_sample_data.py`: Sample data generation utility

### Models
Individual components for specific evaluation aspects:
- `novelty_detector.py`: Detects novelty in proposals using FAISS
- `technical_feasibility.py`: Evaluates technical feasibility
- `financial_viability.py`: Assesses financial aspects
- `risk_ethics_ip.py`: Checks for risks, ethics, and IP issues
- `explainability.py`: Provides model explanations using SHAP/LIME
- `model_monitoring.py`: Monitors model performance and drift

### Utils
Utility functions and helper modules:
- `document_parser.py`: Parses PDF and DOCX documents
- `pdf_generator.py`: Generates PDF reports
- `security.py`: Security management features

### API
REST API endpoints for the system:
- `api.py`: Basic API endpoints
- `api_enhanced.py`: Enhanced API with more features
- `api_secure.py`: Secure API with authentication
- `simple_api.py`: Simplified API for basic operations

### Database
Database management module:
- `database.py`: Database interface and schema management

### Tests
Unit tests for all components:
- `test_*.py`: Individual test files for each component

### Demos
Demonstration and example scripts:
- `demo_*.py`: Demonstration scripts for each component
- `quick_demo.py`: Quick demonstration of the system
- `system_test.py`: System testing script
- `debug_test.py`: Debugging utilities

### Docs
Documentation and deployment files:
- `README.md` and variants: System documentation
- `requirements.txt`: Python dependencies
- `Dockerfile` and `docker-compose.yml`: Containerization files

## Usage

To run the Streamlit application:
```bash
streamlit run core/app_hil.py
```

To run the API server:
```bash
python api/api_secure.py
```

To run tests:
```bash
python -m pytest tests/
```