import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

class ModelExplainer:
    """
    A class to provide explainability for R&D proposal evaluation models using SHAP and LIME
    """
    
    def __init__(self):
        """
        Initialize the Model Explainer
        """
        self.shap_explainers = {}
        self.lime_explainers = {}
        self.feature_names = []
        self.training_data = None
    
    def prepare_training_data(self, proposals_df: pd.DataFrame) -> np.ndarray:
        """
        Prepare training data for explainability
        
        Args:
            proposals_df (pd.DataFrame): DataFrame with proposal data
            
        Returns:
            np.ndarray: Prepared feature matrix
        """
        # Define feature columns
        feature_columns = [
            'Novelty_Score', 'Financial_Score', 'Technical_Score',
            'Coal_Relevance_Score', 'Alignment_Score', 'Clarity_Score',
            'Impact_Score', 'Feasibility_Score'
        ]
        
        # Filter to only include available columns
        available_columns = [col for col in feature_columns if col in proposals_df.columns]
        
        # Store feature names
        self.feature_names = available_columns
        
        # Extract features
        X = proposals_df[available_columns].values
        
        # Store training data for LIME
        self.training_data = X
        
        return X
    
    def train_shap_explainers(self, models: Dict[str, Any], X: np.ndarray) -> None:
        """
        Train SHAP explainers for models
        
        Args:
            models (Dict[str, Any]): Dictionary of trained models
            X (np.ndarray): Training data
        """
        # For tree models (XGBoost, LightGBM, etc.)
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict'):  # Tree-based models
                    explainer = shap.TreeExplainer(model)
                    self.shap_explainers[model_name] = explainer
                elif hasattr(model, 'predict_proba'):  # Linear models
                    explainer = shap.LinearExplainer(model, X)
                    self.shap_explainers[model_name] = explainer
            except Exception as e:
                print(f"Warning: Could not create SHAP explainer for {model_name}: {e}")
    
    def train_lime_explainer(self, X: np.ndarray) -> None:
        """
        Train LIME explainer
        
        Args:
            X (np.ndarray): Training data
        """
        try:
            self.lime_explainers['proposal_evaluator'] = lime.lime_tabular.LimeTabularExplainer(
                X,
                feature_names=self.feature_names,
                class_names=['Not Recommended', 'Recommended'],
                mode='regression'  # Since we're predicting scores
            )
        except Exception as e:
            print(f"Warning: Could not create LIME explainer: {e}")
    
    def explain_with_shap(self, model_name: str, X_instance: np.ndarray) -> Dict[str, Any]:
        """
        Explain a prediction using SHAP
        
        Args:
            model_name (str): Name of the model to use
            X_instance (np.ndarray): Instance to explain
            
        Returns:
            Dict[str, Any]: SHAP explanation
        """
        if model_name not in self.shap_explainers:
            return {"error": f"No SHAP explainer found for model {model_name}"}
        
        try:
            explainer = self.shap_explainers[model_name]
            shap_values = explainer.shap_values(X_instance)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature_name in enumerate(self.feature_names):
                feature_importance[feature_name] = float(shap_values[0][i] if shap_values.ndim > 1 else shap_values[i])
            
            return {
                "shap_values": feature_importance,
                "base_value": float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.0,
                "instance_values": X_instance.tolist()[0] if X_instance.ndim > 1 else X_instance.tolist()
            }
        except Exception as e:
            return {"error": f"Error in SHAP explanation: {e}"}
    
    def explain_with_lime(self, model_predict_fn, X_instance: np.ndarray, num_features: int = 10) -> Dict[str, Any]:
        """
        Explain a prediction using LIME
        
        Args:
            model_predict_fn: Function that takes input and returns predictions
            X_instance (np.ndarray): Instance to explain
            num_features (int): Number of features to include in explanation
            
        Returns:
            Dict[str, Any]: LIME explanation
        """
        if 'proposal_evaluator' not in self.lime_explainers:
            return {"error": "No LIME explainer found"}
        
        try:
            explainer = self.lime_explainers['proposal_evaluator']
            
            # Create explanation
            explanation = explainer.explain_instance(
                X_instance[0] if X_instance.ndim > 1 else X_instance,
                model_predict_fn,
                num_features=num_features
            )
            
            # Extract feature weights
            feature_weights = explanation.as_list()
            
            return {
                "lime_weights": feature_weights,
                "local_prediction": explanation.local_pred[0] if hasattr(explanation, 'local_pred') else 0.0,
                "intercept": explanation.intercept[0] if hasattr(explanation, 'intercept') else 0.0
            }
        except Exception as e:
            return {"error": f"Error in LIME explanation: {e}"}
    
    def get_feature_importance(self, model, X: np.ndarray) -> Dict[str, float]:
        """
        Get feature importance from a model
        
        Args:
            model: Trained model
            X (np.ndarray): Training data
            
        Returns:
            Dict[str, float]: Feature importance scores
        """
        try:
            # For tree-based models
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                feature_importance = {}
                for i, feature_name in enumerate(self.feature_names):
                    feature_importance[feature_name] = float(importance_scores[i])
                return feature_importance
            
            # For linear models
            elif hasattr(model, 'coef_'):
                coef_scores = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
                feature_importance = {}
                for i, feature_name in enumerate(self.feature_names):
                    feature_importance[feature_name] = float(coef_scores[i])
                return feature_importance
            
            else:
                return {"error": "Model does not support feature importance extraction"}
        except Exception as e:
            return {"error": f"Error extracting feature importance: {e}"}
    
    def highlight_text_spans(self, text: str, important_terms: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Highlight important text spans that influenced the score
        
        Args:
            text (str): Text to analyze
            important_terms (List[str]): Important terms to highlight
            top_k (int): Number of top terms to highlight
            
        Returns:
            List[Dict[str, Any]]: Highlighted text spans
        """
        # Get top K important terms
        top_terms = important_terms[:top_k] if len(important_terms) > top_k else important_terms
        
        highlighted_spans = []
        for term in top_terms:
            # Find all occurrences of the term
            start = 0
            while True:
                pos = text.lower().find(term.lower(), start)
                if pos == -1:
                    break
                
                highlighted_spans.append({
                    "text": text[pos:pos+len(term)],
                    "start": pos,
                    "end": pos + len(term),
                    "term": term
                })
                start = pos + 1
        
        return highlighted_spans
    
    def generate_explanation_report(self, proposal_data: Dict[str, Any], 
                                  shap_explanation: Dict[str, Any],
                                  lime_explanation: Dict[str, Any],
                                  feature_importance: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate a comprehensive explanation report
        
        Args:
            proposal_data (Dict[str, Any]): Proposal data
            shap_explanation (Dict[str, Any]): SHAP explanation
            lime_explanation (Dict[str, Any]): LIME explanation
            feature_importance (Dict[str, float]): Feature importance
            
        Returns:
            Dict[str, Any]: Comprehensive explanation report
        """
        # Extract top positive and negative SHAP features
        shap_features = shap_explanation.get("shap_values", {})
        positive_features = {k: v for k, v in shap_features.items() if v > 0}
        negative_features = {k: v for k, v in shap_features.items() if v < 0}
        
        # Sort by absolute value
        top_positive = sorted(positive_features.items(), key=lambda x: x[1], reverse=True)[:3]
        top_negative = sorted(negative_features.items(), key=lambda x: x[1])[:3]
        
        # Extract LIME features
        lime_features = lime_explanation.get("lime_weights", [])
        top_lime = lime_features[:5] if len(lime_features) > 5 else lime_features
        
        # Extract important terms from abstract for text highlighting
        abstract = proposal_data.get("Abstract", "")
        important_terms = [term for term, _ in top_positive]
        highlighted_spans = self.highlight_text_spans(abstract, important_terms)
        
        return {
            "proposal_id": proposal_data.get("Proposal_ID", "Unknown"),
            "proposal_title": proposal_data.get("Title", "Unknown"),
            "overall_score": proposal_data.get("Overall_Score", 0),
            "shap_explanation": {
                "top_positive_contributors": top_positive,
                "top_negative_contributors": top_negative,
                "base_value": shap_explanation.get("base_value", 0)
            },
            "lime_explanation": {
                "top_features": top_lime,
                "local_prediction": lime_explanation.get("local_prediction", 0)
            },
            "feature_importance": feature_importance,
            "highlighted_text_spans": highlighted_spans,
            "interpretation": self._generate_interpretation(top_positive, top_negative, 
                                                          proposal_data.get("Overall_Score", 0))
        }
    
    def _generate_interpretation(self, top_positive: List[Tuple[str, float]], 
                               top_negative: List[Tuple[str, float]], 
                               overall_score: float) -> str:
        """
        Generate human-readable interpretation of the explanation
        
        Args:
            top_positive (List[Tuple[str, float]]): Top positive contributing features
            top_negative (List[Tuple[str, float]]): Top negative contributing features
            overall_score (float): Overall proposal score
            
        Returns:
            str: Human-readable interpretation
        """
        interpretation = f"This proposal has an overall score of {overall_score:.2f} out of 100. "
        
        if top_positive:
            top_pos_feature = top_positive[0][0]
            interpretation += f"The strongest positive contributor was {top_pos_feature}. "
        
        if top_negative:
            top_neg_feature = top_negative[0][0]
            interpretation += f"The main factor reducing the score was {top_neg_feature}. "
        
        if overall_score >= 85:
            interpretation += "This is a highly rated proposal with strong overall merit."
        elif overall_score >= 70:
            interpretation += "This proposal is well-rated but has some areas for improvement."
        else:
            interpretation += "This proposal has significant areas that need improvement."
        
        return interpretation

# Simple mock models for demonstration
class MockXGBoostModel:
    """Mock XGBoost model for demonstration"""
    def __init__(self):
        self.feature_importances_ = np.array([0.2, 0.15, 0.25, 0.1, 0.1, 0.05, 0.1, 0.05])
    
    def predict(self, X):
        # Simple mock prediction
        return np.sum(X * self.feature_importances_, axis=1)

class MockLightGBMModel:
    """Mock LightGBM model for demonstration"""
    def __init__(self):
        self.feature_importances_ = np.array([0.15, 0.2, 0.1, 0.25, 0.05, 0.1, 0.1, 0.05])
    
    def predict(self, X):
        # Simple mock prediction
        return np.sum(X * self.feature_importances_, axis=1)

def create_sample_proposals() -> pd.DataFrame:
    """
    Create sample proposal data for demonstration
    
    Returns:
        pd.DataFrame: Sample proposal data
    """
    return pd.DataFrame({
        'Proposal_ID': ['PROP001', 'PROP002', 'PROP003', 'PROP004', 'PROP005'],
        'Title': [
            'AI for Mine Safety Monitoring',
            'Coal Quality Prediction Using Machine Learning',
            'Automated Coal Washing Process Optimization',
            'Methane Gas Detection and Alert System',
            'Sustainable Rehabilitation of Mining Areas'
        ],
        'Abstract': [
            'Using artificial intelligence to monitor and predict safety hazards in coal mines.',
            'Developing machine learning models to predict coal quality based on geological data.',
            'Implementing automated control systems to optimize the coal washing process.',
            'Creating a real-time methane gas detection system with automated alerts.',
            'Developing sustainable methods for rehabilitating mining areas with native vegetation.'
        ],
        'Novelty_Score': [0.85, 0.75, 0.80, 0.90, 0.70],
        'Financial_Score': [0.75, 0.85, 0.80, 0.70, 0.90],
        'Technical_Score': [0.90, 0.85, 0.88, 0.92, 0.75],
        'Coal_Relevance_Score': [0.95, 0.90, 0.85, 0.92, 0.88],
        'Alignment_Score': [0.88, 0.85, 0.90, 0.87, 0.92],
        'Clarity_Score': [0.82, 0.90, 0.85, 0.88, 0.80],
        'Impact_Score': [0.90, 0.80, 0.85, 0.95, 0.85],
        'Feasibility_Score': [0.88, 0.85, 0.90, 0.92, 0.78],
        'Overall_Score': [86.5, 83.2, 86.8, 89.1, 82.4]
    })

def mock_predict_fn(X):
    """
    Mock prediction function for LIME
    
    Args:
        X: Input features
        
    Returns:
        Predictions
    """
    # Simple linear combination for mock prediction
    weights = np.array([0.2, 0.15, 0.25, 0.1, 0.1, 0.05, 0.1, 0.05])
    if X.ndim == 1:
        return np.sum(X * weights)
    else:
        return np.sum(X * weights, axis=1)

def main():
    """
    Main function to demonstrate the Model Explainer
    """
    print("Demonstrating Model Explainer with SHAP and LIME")
    print("=" * 50)
    
    # Create explainer
    explainer = ModelExplainer()
    print("✓ Model Explainer initialized")
    
    # Create sample data
    sample_proposals = create_sample_proposals()
    print(f"✓ Sample data created with {len(sample_proposals)} proposals")
    
    # Prepare training data
    X_train = explainer.prepare_training_data(sample_proposals)
    print(f"✓ Training data prepared: {X_train.shape}")
    
    # Create mock models
    models = {
        'xgboost': MockXGBoostModel(),
        'lightgbm': MockLightGBMModel()
    }
    
    # Train explainers
    explainer.train_shap_explainers(models, X_train)
    explainer.train_lime_explainer(X_train)
    print("✓ SHAP and LIME explainers trained")
    
    # Explain a sample proposal
    sample_proposal = sample_proposals.iloc[0]
    X_instance = X_train[0:1]  # First proposal as instance to explain
    
    print(f"\nExplaining proposal: {sample_proposal['Title']}")
    
    # SHAP explanation
    shap_explanation = explainer.explain_with_shap('xgboost', X_instance)
    print(f"✓ SHAP explanation generated")
    
    # LIME explanation
    lime_explanation = explainer.explain_with_lime(mock_predict_fn, X_instance)
    print(f"✓ LIME explanation generated")
    
    # Feature importance
    feature_importance = explainer.get_feature_importance(models['xgboost'], X_train)
    print(f"✓ Feature importance extracted")
    
    # Generate comprehensive report
    report = explainer.generate_explanation_report(
        sample_proposal.to_dict(),
        shap_explanation,
        lime_explanation,
        feature_importance
    )
    
    # Display key findings
    print(f"\nProposal: {report['proposal_title']}")
    print(f"Overall Score: {report['overall_score']:.2f}")
    print(f"Interpretation: {report['interpretation']}")
    
    print(f"\nTop Positive Contributors (SHAP):")
    for feature, contribution in report['shap_explanation']['top_positive_contributors']:
        print(f"  {feature}: +{contribution:.4f}")
    
    print(f"\nTop Negative Contributors (SHAP):")
    for feature, contribution in report['shap_explanation']['top_negative_contributors']:
        print(f"  {feature}: {contribution:.4f}")
    
    print(f"\nFeature Importance:")
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_importance[:5]:
        print(f"  {feature}: {importance:.4f}")
    
    print("\n✓ Model Explainer demonstration completed successfully")

if __name__ == "__main__":
    main()