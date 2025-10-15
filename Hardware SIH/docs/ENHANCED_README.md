# Enhanced AI/ML Based Auto Evaluation System for R&D Proposals

An enhanced AI/ML system to automatically evaluate R&D proposals based on novelty, funding requirements, technical feasibility, and coal industry relevance.

## Features

1. **Novelty Scoring**: Uses Sentence-BERT embeddings or TF-IDF to compute semantic similarity between new proposals and past proposals
2. **Funding Scoring**: Normalizes funding requests and gives higher scores to lower funding proposals
3. **Feasibility Scoring**: Evaluates technical feasibility based on keyword matching
4. **Coal Relevance Scoring**: Evaluates relevance to coal industry based on domain-specific keyword matching
5. **Weighted Evaluation**: Combines scores using configurable weights
6. **Ranking**: Ranks proposals by their final evaluation score
7. **Multiple Model Support**: Supports both Sentence-BERT and TF-IDF based similarity calculation
8. **Web Interface**: Streamlit-based UI for easy interaction
9. **RESTful API**: Flask-based API for integration with other systems
10. **Export Results**: Generates CSV reports of evaluated proposals

## Technology Stack

### Programming Languages
- **Python** – Primary language for AI/ML development, data processing, and automation

### AI/ML & Data Processing Tools
- **Scikit-learn** – For classical ML models like classification, regression, clustering
- **XGBoost** – For advanced scoring or ranking models
- **Sentence-BERT** – For semantic comparison with past projects
- **NLTK** – For preprocessing, tokenization, and entity recognition
- **TF-IDF** – Alternative text similarity calculation method

### Database & Storage
- **CSV Files** – For structured storage of proposals and evaluation data (can be extended to SQL/NoSQL)

### Web & UI Tools
- **Streamlit** – For interactive web interface
- **Flask** – For RESTful API backend
- **Matplotlib/Seaborn** – For dashboards showing proposal scores and trends

### Automation & Workflow
- **Flask** – For scheduling automated proposal evaluation workflows

### Analytics & Reporting
- **Matplotlib, Seaborn** – For dashboards showing proposal scores and trends
- **CSV generation** – For automated evaluation reports

### Cloud / DevOps
- **Docker** – To package ML models and apps (Dockerfile included)
- **Git** – Version control

## Installation

1. Clone or download this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

Run the enhanced evaluation script directly:
```bash
python test_enhanced.py
```

To use with your own data:
1. Prepare two CSV files:
   - `past_proposals.csv`: Contains historical proposals with columns [Proposal_ID, Title, Abstract, Funding_Requested]
   - `new_proposals.csv`: Contains proposals to evaluate with the same columns
2. Modify the `main()` function in `enhanced_evaluator.py` to use your file paths
3. Run the script

### Web Interface

Launch the Streamlit web app:
```bash
streamlit run app.py
```

### RESTful API

Launch the Flask API:
```bash
python api.py
```

API Endpoints:
- `GET /health` - Health check endpoint
- `POST /evaluate` - Evaluate proposals
- `POST /sample-data` - Generate sample data
- `POST /download-results` - Download evaluation results as CSV

Example API usage:
```bash
curl -X POST http://localhost:5000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "past_proposals_path": "sample_past_proposals.csv",
    "new_proposals_path": "sample_new_proposals.csv",
    "include_feasibility": true,
    "model_type": "sentence_bert"
  }'
```

## How It Works

### Novelty Score
- Uses Sentence-BERT (`all-MiniLM-L6-v2`) or TF-IDF to generate embeddings for proposal abstracts
- Computes cosine similarity between new proposals and all past proposals
- Novelty score = 1 - maximum similarity (higher means more novel)

### Funding Score
- Normalizes all funding requests using Min-Max scaling
- Inverts the normalized value so lower funding gets higher scores
- Funding score = 1 - normalized_funding

### Feasibility Score
- Counts matches of technical keywords in proposal title and abstract
- Normalizes by total number of keywords
- Keywords include: algorithm, data, model, framework, platform, etc.

### Coal Relevance Score
- Counts matches of coal industry specific keywords in proposal title and abstract
- Normalizes by total number of coal keywords
- Keywords include: coal, mining, extraction, safety, ventilation, gas, etc.

### Final Evaluation Score
Default weighting (with feasibility scoring enabled):
- Novelty: 40%
- Funding: 20%
- Feasibility: 20%
- Coal Relevance: 20%

With feasibility scoring disabled:
- Novelty: 50%
- Funding: 25%
- Coal Relevance: 25%

## File Structure

- `enhanced_evaluator.py`: Enhanced core evaluation engine with multiple model support
- `proposal_evaluator.py`: Original evaluation engine
- `app.py`: Streamlit web interface
- `api.py`: Flask RESTful API
- `requirements.txt`: Python package dependencies
- `sample_past_proposals.csv`: Example historical proposals
- `sample_new_proposals.csv`: Example proposals to evaluate

## Docker Support

A Dockerfile is included to containerize the application:

```bash
docker build -t rdp-evaluator .
docker run -p 8501:8501 -p 5000:5000 rdp-evaluator
```

## Customization

You can customize:
1. Technical keywords for feasibility scoring (modify `technical_keywords` in `EnhancedRDPEvaluator` class)
2. Coal industry keywords for relevance scoring (modify `calculate_coal_relevance_score` method)
3. Weighting scheme (adjust weights in web interface or modify default values)
4. Text embedding model (change model name in `EnhancedRDPEvaluator.__init__`)
5. Text preprocessing steps (modify `preprocess_text` method)
6. Model type (choose between 'sentence_bert' and 'tfidf')

## Security & Compliance

- Authentication can be added to the Flask API using OAuth2/JWT tokens
- Data encryption can be implemented for sensitive proposal data
- SSL/TLS can be configured for secure communication

## Performance & Scalability

- The system can be deployed on cloud platforms (AWS/Azure/GCP) for scalability
- Models can be optimized for performance using techniques like quantization
- Caching can be implemented for frequently accessed data
- Load balancing can be configured for high-traffic scenarios

## License

This project is open source and available under the MIT License.