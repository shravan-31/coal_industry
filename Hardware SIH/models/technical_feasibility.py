import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline
import torch
import re
from typing import List, Dict, Tuple

class TechnicalFeasibilityEvaluator:
    """
    A class to evaluate technical feasibility of R&D proposals using ensemble methods
    """
    
    def __init__(self):
        """
        Initialize the Technical Feasibility Evaluator
        """
        self.xgb_model = None
        self.lgb_model = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.technical_keywords = [
            'algorithm', 'data', 'model', 'framework', 'platform', 'system',
            'network', 'database', 'api', 'interface', 'protocol', 'security',
            'scalability', 'performance', 'optimization', 'machine learning',
            'artificial intelligence', 'neural network', 'deep learning',
            'computer vision', 'natural language processing', 'blockchain',
            'cloud', 'iot', 'internet of things', 'big data', 'analytics',
            'software', 'hardware', 'implementation', 'design', 'development',
            'testing', 'validation', 'experiment', 'research', 'study'
        ]
        
        # Numerical feature columns
        self.numerical_features = [
            'team_experience_score', 
            'budget_realism_score', 
            'timeline_realism_score',
            'methodology_completeness_score'
        ]
    
    def extract_numerical_features(self, proposals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract numerical features from proposals
        
        Args:
            proposals_df (pd.DataFrame): DataFrame with proposal data
            
        Returns:
            pd.DataFrame: DataFrame with extracted numerical features
        """
        features = pd.DataFrame()
        
        # Team experience score (based on keywords in title/abstract)
        features['team_experience_score'] = proposals_df.apply(
            lambda row: self._calculate_team_experience(row['Title'], row['Abstract']), axis=1
        )
        
        # Budget realism score (placeholder - would need actual budget data)
        features['budget_realism_score'] = proposals_df.apply(
            lambda row: self._calculate_budget_realism(row), axis=1
        )
        
        # Timeline realism score (placeholder - would need actual timeline data)
        features['timeline_realism_score'] = proposals_df.apply(
            lambda row: self._calculate_timeline_realism(row), axis=1
        )
        
        # Methodology completeness score
        features['methodology_completeness_score'] = proposals_df.apply(
            lambda row: self._calculate_methodology_completeness(row['Abstract']), axis=1
        )
        
        return features
    
    def _calculate_team_experience(self, title: str, abstract: str) -> float:
        """
        Calculate team experience score based on keywords
        
        Args:
            title (str): Proposal title
            abstract (str): Proposal abstract
            
        Returns:
            float: Team experience score (0-1)
        """
        text = (str(title) + " " + str(abstract)).lower()
        
        # Experience indicators
        experience_keywords = [
            'phd', 'doctorate', 'masters', 'ms', 'experience', 'expert', 
            'proven', 'track record', 'successful', 'published', 'research',
            'development', 'implementation', 'deployed', 'production'
        ]
        
        matched_keywords = sum(1 for keyword in experience_keywords if keyword in text)
        return min(matched_keywords / len(experience_keywords), 1.0)
    
    def _calculate_budget_realism(self, proposal_row) -> float:
        """
        Calculate budget realism score (placeholder implementation)
        
        Args:
            proposal_row: Row from proposals DataFrame
            
        Returns:
            float: Budget realism score (0-1)
        """
        # Placeholder implementation - in a real system, this would analyze budget details
        # For now, we'll use a simple heuristic based on funding requested
        funding = proposal_row.get('Funding_Requested', 0)
        if funding <= 0:
            return 0.5  # Neutral score if no funding info
        
        # Assume reasonable budget range is $10K-$500K
        if funding < 10000:
            return 0.3  # Too low
        elif funding > 500000:
            return 0.4  # Too high
        else:
            return 0.8  # Reasonable range
    
    def _calculate_timeline_realism(self, proposal_row) -> float:
        """
        Calculate timeline realism score (placeholder implementation)
        
        Args:
            proposal_row: Row from proposals DataFrame
            
        Returns:
            float: Timeline realism score (0-1)
        """
        # Placeholder implementation - in a real system, this would analyze timeline details
        return 0.7  # Default score
    
    def _calculate_methodology_completeness(self, abstract: str) -> float:
        """
        Calculate methodology completeness score based on presence of key elements
        
        Args:
            abstract (str): Proposal abstract
            
        Returns:
            float: Methodology completeness score (0-1)
        """
        if not abstract:
            return 0.0
            
        text = abstract.lower()
        
        # Methodology elements
        methodology_elements = [
            'method', 'approach', 'technique', 'procedure', 'process',
            'experiment', 'study', 'analysis', 'design', 'implementation',
            'testing', 'validation', 'evaluation', 'data collection',
            'hypothesis', 'objective', 'goal'
        ]
        
        matched_elements = sum(1 for element in methodology_elements if element in text)
        return min(matched_elements / len(methodology_elements), 1.0)
    
    def prepare_text_features(self, texts: List[str]) -> np.ndarray:
        """
        Prepare text features using TF-IDF
        
        Args:
            texts (List[str]): List of texts to vectorize
            
        Returns:
            np.ndarray: TF-IDF features
        """
        return self.tfidf_vectorizer.fit_transform(texts).toarray()
    
    def train_xgboost_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train XGBoost model for feasibility classification
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels
        """
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.xgb_model.fit(X, y)
    
    def train_lightgbm_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train LightGBM model for feasibility classification
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels
        """
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.lgb_model.fit(X, y)
    
    def train_bert_model(self, texts: List[str], labels: List[int]) -> None:
        """
        Train BERT model for feasibility classification (simplified implementation)
        
        Args:
            texts (List[str]): List of texts
            labels (List[int]): List of labels
        """
        # In a real implementation, we would fine-tune a BERT model
        # For this implementation, we'll use a pre-trained classifier
        self.bert_classifier = pipeline(
            "text-classification", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            return_all_scores=True
        )
    
    def predict_feasibility_xgboost(self, X: np.ndarray) -> np.ndarray:
        """
        Predict feasibility using XGBoost model
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predictions
        """
        if self.xgb_model is None:
            raise ValueError("XGBoost model not trained yet")
        return self.xgb_model.predict_proba(X)[:, 1]  # Return probability of positive class
    
    def predict_feasibility_lightgbm(self, X: np.ndarray) -> np.ndarray:
        """
        Predict feasibility using LightGBM model
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predictions
        """
        if self.lgb_model is None:
            raise ValueError("LightGBM model not trained yet")
        return self.lgb_model.predict_proba(X)[:, 1]  # Return probability of positive class
    
    def predict_feasibility_bert(self, texts: List[str]) -> np.ndarray:
        """
        Predict feasibility using BERT model
        
        Args:
            texts (List[str]): List of texts
            
        Returns:
            np.ndarray: Predictions
        """
        if self.bert_classifier is None:
            raise ValueError("BERT classifier not initialized")
        
        # For simplicity, we'll use a heuristic based on sentiment analysis
        # In a real implementation, we would have a domain-specific model
        predictions = []
        for text in texts:
            # Truncate text to avoid memory issues
            if len(text) > 512:
                text = text[:512]
            
            try:
                result = self.bert_classifier(text)
                # Use positive sentiment as a proxy for feasibility
                pos_score = next(item['score'] for item in result[0] if item['label'] == 'POSITIVE')
                predictions.append(pos_score)
            except:
                predictions.append(0.5)  # Default neutral score
        
        return np.array(predictions)
    
    def ensemble_predict(self, X: np.ndarray, texts: List[str]) -> np.ndarray:
        """
        Make ensemble predictions using all models
        
        Args:
            X (np.ndarray): Feature matrix
            texts (List[str]): List of texts
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        # Get predictions from all models
        xgb_pred = self.predict_feasibility_xgboost(X)
        lgb_pred = self.predict_feasibility_lightgbm(X)
        bert_pred = self.predict_feasibility_bert(texts)
        
        # Simple averaging ensemble
        ensemble_pred = (xgb_pred + lgb_pred + bert_pred) / 3
        return ensemble_pred
    
    def evaluate_proposal(self, proposal_data: Dict) -> Dict:
        """
        Evaluate a single proposal for technical feasibility
        
        Args:
            proposal_data (Dict): Proposal data including Title and Abstract
            
        Returns:
            Dict: Feasibility scores and details
        """
        # Extract features
        numerical_features = self.extract_numerical_features(
            pd.DataFrame([proposal_data])
        ).values[0]
        
        # Prepare text
        text = f"{proposal_data.get('Title', '')} {proposal_data.get('Abstract', '')}"
        
        # Calculate component scores
        team_score = numerical_features[0]
        budget_score = numerical_features[1]
        timeline_score = numerical_features[2]
        methodology_score = numerical_features[3]
        
        # Use BERT prediction as overall feasibility proxy
        bert_score = self.predict_feasibility_bert([text])[0]
        
        # Calculate final feasibility score (weighted average)
        feasibility_score = (
            0.25 * team_score +
            0.20 * budget_score +
            0.15 * timeline_score +
            0.20 * methodology_score +
            0.20 * bert_score
        )
        
        return {
            'feasibility_score': feasibility_score,
            'team_experience_score': team_score,
            'budget_realism_score': budget_score,
            'timeline_realism_score': timeline_score,
            'methodology_completeness_score': methodology_score,
            'bert_analysis_score': bert_score
        }

def create_sample_training_data() -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create sample training data for demonstration
    
    Returns:
        Tuple[pd.DataFrame, np.ndarray]: (features DataFrame, labels array)
    """
    # Sample data
    data = {
        'Proposal_ID': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P010'],
        'Title': [
            'Machine Learning for Image Recognition',
            'Cloud Computing Infrastructure',
            'Data Mining Techniques',
            'Natural Language Processing Framework',
            'IoT Security Solutions',
            'Vague Research Idea',
            'Unrealistic Project Plan',
            'Incomplete Methodology',
            'Advanced AI for Coal Mining',
            'Blockchain for Supply Chain'
        ],
        'Abstract': [
            'This project focuses on developing advanced machine learning algorithms for image recognition tasks with a detailed methodology and experienced team.',
            'Research on scalable cloud computing infrastructure for enterprise applications with clear objectives and realistic timeline.',
            'Exploring data mining techniques for extracting valuable insights from large datasets with proven approaches.',
            'Development of a comprehensive framework for natural language processing applications with extensive testing plans.',
            'Investigating security challenges and solutions for Internet of Things devices with comprehensive risk assessment.',
            'Some research idea without clear objectives or methodology.',
            'A project that promises to solve all world problems in one month with zero budget.',
            'Research on something without any details on how it will be done.',
            'Using advanced AI techniques to optimize coal mining operations with detailed technical approach and experienced team.',
            'Applying blockchain technology to enhance transparency in supply chains with clear implementation plan.'
        ],
        'Funding_Requested': [50000, 75000, 30000, 60000, 45000, 10000, 0, 20000, 80000, 70000]
    }
    
    df = pd.DataFrame(data)
    
    # Sample labels (1 = feasible, 0 = not feasible)
    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 1, 1])
    
    return df, labels

def main():
    """
    Main function to demonstrate the Technical Feasibility Evaluator
    """
    # Create evaluator
    evaluator = TechnicalFeasibilityEvaluator()
    
    # Create sample data
    df, labels = create_sample_training_data()
    
    # Extract features
    numerical_features = evaluator.extract_numerical_features(df)
    text_features = evaluator.prepare_text_features(df['Abstract'].tolist())
    
    # Combine features
    X = np.hstack([numerical_features.values, text_features])
    
    # Train models (in a real scenario, we would have more data)
    print("Training models...")
    try:
        evaluator.train_xgboost_model(X, labels)
        evaluator.train_lightgbm_model(X, labels)
        evaluator.train_bert_model(df['Abstract'].tolist(), labels)
        print("Models trained successfully!")
    except Exception as e:
        print(f"Error training models: {e}")
        return
    
    # Evaluate sample proposals
    sample_proposals = [
        {
            'Title': 'AI for Mine Safety Monitoring',
            'Abstract': 'Using artificial intelligence to monitor and predict safety hazards in coal mines to prevent accidents with a detailed technical approach.',
            'Funding_Requested': 80000
        },
        {
            'Title': 'Vague Research Concept',
            'Abstract': 'Some research without clear methodology or objectives.',
            'Funding_Requested': 10000
        }
    ]
    
    print("\nEvaluating sample proposals:")
    for i, proposal in enumerate(sample_proposals, 1):
        print(f"\nProposal {i}: {proposal['Title']}")
        result = evaluator.evaluate_proposal(proposal)
        print(f"Feasibility Score: {result['feasibility_score']:.4f}")
        print(f"Team Experience: {result['team_experience_score']:.4f}")
        print(f"Budget Realism: {result['budget_realism_score']:.4f}")
        print(f"Methodology Completeness: {result['methodology_completeness_score']:.4f}")
        print(f"BERT Analysis: {result['bert_analysis_score']:.4f}")

if __name__ == "__main__":
    main()