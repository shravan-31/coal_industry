import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
import re
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class RDPEvaluator:
    def __init__(self, model_type='sentence_bert'):
        """
        Initialize the R&D Proposal Evaluator
        
        Args:
            model_type (str): Type of model to use for text embedding ('sentence_bert', 'tfidf')
        """
        self.model_type = model_type
        
        # Load the sentence transformer model for semantic similarity
        if model_type == 'sentence_bert':
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        elif model_type == 'tfidf':
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        self.stop_words = set(stopwords.words('english'))
        
        # Weights for combining scores
        self.novelty_weight = 0.4
        self.funding_weight = 0.2
        self.feasibility_weight = 0.2
        self.coal_relevance_weight = 0.2
        
        # Technical keywords for feasibility scoring (can be expanded)
        self.technical_keywords = [
            'algorithm', 'data', 'model', 'framework', 'platform', 'system',
            'network', 'database', 'api', 'interface', 'protocol', 'security',
            'scalability', 'performance', 'optimization', 'machine learning',
            'artificial intelligence', 'neural network', 'deep learning',
            'computer vision', 'natural language processing', 'blockchain',
            'cloud', 'iot', 'internet of things', 'big data', 'analytics',
            # Coal industry specific terms
            'coal', 'mining', 'extraction', 'geological', 'seismic', 'exploration',
            'safety', 'ventilation', 'gas', 'methane', 'detection', 'monitoring',
            'processing', 'cleaning', 'washing', 'beneficiation', 'transport',
            'combustion', 'emissions', 'environmental', 'reclamation', 'rehabilitation',
            'equipment', 'machinery', 'automation', 'control', 'sensors', 'instrumentation'
        ]
    
    def load_data(self, past_proposals_path, new_proposals_path):
        """
        Load past and new proposals from CSV files
        
        Args:
            past_proposals_path (str): Path to past proposals CSV
            new_proposals_path (str): Path to new proposals CSV
            
        Returns:
            tuple: (past_proposals_df, new_proposals_df)
        """
        past_proposals = pd.read_csv(past_proposals_path)
        new_proposals = pd.read_csv(new_proposals_path)
        
        return past_proposals, new_proposals
    
    def preprocess_text(self, text):
        """
        Preprocess text by removing special characters and extra spaces
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def calculate_novelty_score(self, new_abstract, past_abstracts):
        """
        Calculate novelty score using semantic similarity
        
        Args:
            new_abstract (str): Abstract of the new proposal
            past_abstracts (list): List of abstracts from past proposals
            
        Returns:
            float: Novelty score (higher means more novel)
        """
        # Preprocess texts
        new_abstract = self.preprocess_text(new_abstract)
        past_abstracts = [self.preprocess_text(abst) for abst in past_abstracts]
        
        if self.model_type == 'sentence_bert':
            # Get embeddings using Sentence-BERT
            new_embedding = self.model.encode([new_abstract])
            past_embeddings = self.model.encode(past_abstracts)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(new_embedding, past_embeddings)
        elif self.model_type == 'tfidf':
            # Use TF-IDF for similarity calculation
            # Fit on past abstracts + new abstract
            all_texts = past_abstracts + [new_abstract]
            self.tfidf_vectorizer.fit(all_texts)
            
            # Transform texts
            new_tfidf = self.tfidf_vectorizer.transform([new_abstract])
            past_tfidf = self.tfidf_vectorizer.transform(past_abstracts)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(new_tfidf, past_tfidf)
        
        # Novelty is inverse of maximum similarity (higher similarity = less novel)
        max_similarity = np.max(similarities)
        novelty_score = 1 - max_similarity
        
        return novelty_score
    
    def calculate_funding_score(self, funding_requested, past_fundings):
        """
        Calculate normalized funding score (lower funding = higher score)
        
        Args:
            funding_requested (float): Funding requested for new proposal
            past_fundings (list): List of fundings from past proposals
            
        Returns:
            float: Normalized funding score
        """
        # Combine past fundings with current funding for normalization
        all_fundings = list(past_fundings) + [funding_requested]
        
        # Normalize using Min-Max scaling
        scaler = MinMaxScaler()
        normalized_fundings = scaler.fit_transform(np.array(all_fundings).reshape(-1, 1)).flatten()
        
        # Get the normalized value for the current proposal
        current_normalized = normalized_fundings[-1]
        
        # Since lower funding should get higher score, we invert the normalized value
        funding_score = 1 - current_normalized
        
        return funding_score
    
    def calculate_feasibility_score(self, title, abstract):
        """
        Calculate technical feasibility score based on keyword matching
        
        Args:
            title (str): Title of the proposal
            abstract (str): Abstract of the proposal
            
        Returns:
            float: Feasibility score
        """
        # Combine title and abstract
        full_text = (title + " " + abstract).lower()
        
        # Count matching technical keywords
        matched_keywords = 0
        for keyword in self.technical_keywords:
            if keyword.lower() in full_text:
                matched_keywords += 1
        
        # Normalize score (0 to 1)
        feasibility_score = matched_keywords / len(self.technical_keywords)
        
        return feasibility_score
    
    def calculate_coal_relevance_score(self, title, abstract):
        """
        Calculate coal industry relevance score based on domain-specific keyword matching
        
        Args:
            title (str): Title of the proposal
            abstract (str): Abstract of the proposal
            
        Returns:
            float: Coal relevance score
        """
        # Coal industry specific keywords
        coal_keywords = [
            'coal', 'mining', 'extraction', 'geological', 'seismic', 'exploration',
            'safety', 'ventilation', 'gas', 'methane', 'detection', 'monitoring',
            'processing', 'cleaning', 'washing', 'beneficiation', 'transport',
            'combustion', 'emissions', 'environmental', 'reclamation', 'rehabilitation',
            'equipment', 'machinery', 'automation', 'control', 'sensors', 'instrumentation'
        ]
        
        # Combine title and abstract
        full_text = (title + " " + abstract).lower()
        
        # Count matching coal keywords
        matched_keywords = 0
        for keyword in coal_keywords:
            if keyword.lower() in full_text:
                matched_keywords += 1
        
        # Normalize score (0 to 1)
        coal_relevance_score = matched_keywords / len(coal_keywords)
        
        return coal_relevance_score
    
    def train_advanced_models(self, past_proposals):
        """
        Train advanced ML models for proposal evaluation
        
        Args:
            past_proposals (DataFrame): Historical proposals with evaluation scores
        """
        # This would be implemented if we had historical evaluation data
        # For now, we'll initialize the models
        self.xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        self.lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        
        # In a real implementation, we would train these models on historical data
        # For now, we'll just initialize them for potential future use
    
    def evaluate_proposals(self, past_proposals_path, new_proposals_path, include_feasibility=False):
        """
        Main evaluation function to score and rank proposals
        
        Args:
            past_proposals_path (str): Path to past proposals CSV
            new_proposals_path (str): Path to new proposals CSV
            include_feasibility (bool): Whether to include feasibility scoring
            
        Returns:
            DataFrame: Evaluated proposals with scores
        """
        # Load data
        past_proposals, new_proposals = self.load_data(past_proposals_path, new_proposals_path)
        
        # Extract past abstracts and fundings
        past_abstracts = past_proposals['Abstract'].tolist()
        past_fundings = past_proposals['Funding_Requested'].tolist()
        
        # Lists to store scores
        novelty_scores = []
        funding_scores = []
        feasibility_scores = []
        coal_relevance_scores = []
        evaluation_scores = []
        
        # Evaluate each new proposal
        for index, proposal in new_proposals.iterrows():
            # Calculate novelty score
            novelty_score = self.calculate_novelty_score(proposal['Abstract'], past_abstracts)
            novelty_scores.append(novelty_score)
            
            # Calculate funding score
            funding_score = self.calculate_funding_score(proposal['Funding_Requested'], past_fundings)
            funding_scores.append(funding_score)
            
            # Calculate feasibility score if requested
            if include_feasibility:
                feasibility_score = self.calculate_feasibility_score(proposal['Title'], proposal['Abstract'])
                feasibility_scores.append(feasibility_score)
            else:
                feasibility_score = 0.0
            
            # Calculate coal relevance score
            coal_relevance_score = self.calculate_coal_relevance_score(proposal['Title'], proposal['Abstract'])
            coal_relevance_scores.append(coal_relevance_score)
            
            # Calculate final evaluation score
            if include_feasibility:
                # With feasibility: 40% novelty, 20% funding, 20% feasibility, 20% coal relevance
                evaluation_score = (self.novelty_weight * novelty_score) + \
                                 (self.funding_weight * funding_score) + \
                                 (self.feasibility_weight * feasibility_score) + \
                                 (self.coal_relevance_weight * coal_relevance_score)
            else:
                # Without feasibility: 50% novelty, 25% funding, 25% coal relevance
                evaluation_score = (0.5 * novelty_score) + \
                                 (0.25 * funding_score) + \
                                 (0.25 * coal_relevance_score)
            
            evaluation_scores.append(evaluation_score)
        
        # Add scores to the dataframe
        new_proposals['Novelty_Score'] = novelty_scores
        new_proposals['Funding_Score'] = funding_scores
        new_proposals['Coal_Relevance_Score'] = coal_relevance_scores
        if include_feasibility:
            new_proposals['Feasibility_Score'] = feasibility_scores
        new_proposals['Evaluation_Score'] = evaluation_scores
        
        # Sort by evaluation score (descending)
        new_proposals = new_proposals.sort_values(by='Evaluation_Score', ascending=False)
        
        return new_proposals
    
    def save_results(self, evaluated_proposals, output_path):
        """
        Save evaluated proposals to CSV
        
        Args:
            evaluated_proposals (DataFrame): Evaluated proposals
            output_path (str): Path to save the output CSV
        """
        # Select only the required columns
        if 'Feasibility_Score' in evaluated_proposals.columns:
            output_columns = ['Proposal_ID', 'Title', 'Novelty_Score', 'Funding_Score', 
                            'Feasibility_Score', 'Coal_Relevance_Score', 'Evaluation_Score']
        else:
            output_columns = ['Proposal_ID', 'Title', 'Novelty_Score', 'Funding_Score', 
                            'Coal_Relevance_Score', 'Evaluation_Score']
            
        output_df = evaluated_proposals[output_columns]
        output_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    def display_top_proposals(self, evaluated_proposals, top_n=5):
        """
        Display top N proposals with their scores
        
        Args:
            evaluated_proposals (DataFrame): Evaluated proposals
            top_n (int): Number of top proposals to display
        """
        print(f"\nTop {top_n} Proposals:")
        print("=" * 80)
        
        top_proposals = evaluated_proposals.head(top_n)
        
        for index, proposal in top_proposals.iterrows():
            print(f"Proposal ID: {proposal['Proposal_ID']}")
            print(f"Title: {proposal['Title']}")
            print(f"Novelty Score: {proposal['Novelty_Score']:.4f}")
            print(f"Funding Score: {proposal['Funding_Score']:.4f}")
            if 'Feasibility_Score' in proposal:
                print(f"Feasibility Score: {proposal['Feasibility_Score']:.4f}")
            print(f"Coal Relevance Score: {proposal['Coal_Relevance_Score']:.4f}")
            print(f"Evaluation Score: {proposal['Evaluation_Score']:.4f}")
            print("-" * 80)

def main():
    """
    Main function to demonstrate the evaluator
    """
    # Initialize evaluator
    evaluator = RDPEvaluator()
    
    # Example usage (you would replace these paths with actual file paths)
    # past_proposals_path = 'past_proposals.csv'
    # new_proposals_path = 'new_proposals.csv'
    # output_path = 'evaluated_proposals.csv'
    
    # For demonstration, we'll create sample data
    create_sample_data()
    
    # Evaluate proposals
    print("Evaluating proposals...")
    evaluated_proposals = evaluator.evaluate_proposals(
        'sample_past_proposals.csv', 
        'sample_new_proposals.csv',
        include_feasibility=True
    )
    
    # Save results
    evaluator.save_results(evaluated_proposals, 'evaluated_proposals.csv')
    
    # Display top proposals
    evaluator.display_top_proposals(evaluated_proposals, top_n=3)

def create_sample_data():
    """
    Create sample data for demonstration
    """
    # Sample past proposals
    past_data = {
        'Proposal_ID': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'Title': [
            'Machine Learning for Image Recognition',
            'Cloud Computing Infrastructure',
            'Data Mining Techniques',
            'Natural Language Processing Framework',
            'IoT Security Solutions'
        ],
        'Abstract': [
            'This project focuses on developing advanced machine learning algorithms for image recognition tasks.',
            'Research on scalable cloud computing infrastructure for enterprise applications.',
            'Exploring data mining techniques for extracting valuable insights from large datasets.',
            'Development of a comprehensive framework for natural language processing applications.',
            'Investigating security challenges and solutions for Internet of Things devices.'
        ],
        'Funding_Requested': [50000, 75000, 30000, 60000, 45000]
    }
    
    past_df = pd.DataFrame(past_data)
    past_df.to_csv('sample_past_proposals.csv', index=False)
    
    # Sample new proposals
    new_data = {
        'Proposal_ID': ['N001', 'N002', 'N003', 'N004', 'N005'],
        'Title': [
            'Deep Learning for Medical Imaging',
            'Blockchain for Supply Chain Management',
            'Reinforcement Learning in Robotics',
            'Computer Vision for Autonomous Vehicles',
            'Quantum Computing Algorithms'
        ],
        'Abstract': [
            'Using deep learning techniques to improve medical imaging diagnostics and accuracy.',
            'Applying blockchain technology to enhance transparency and security in supply chains.',
            'Research on reinforcement learning algorithms for adaptive robotic control systems.',
            'Developing computer vision systems for real-time decision making in autonomous vehicles.',
            'Exploring quantum computing algorithms for solving complex computational problems.'
        ],
        'Funding_Requested': [80000, 70000, 65000, 90000, 100000]
    }
    
def create_sample_data():
    """
    Create sample data for demonstration
    """
    # Sample past proposals
    past_data = {
        'Proposal_ID': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'Title': [
            'Machine Learning for Image Recognition',
            'Cloud Computing Infrastructure',
            'Data Mining Techniques',
            'Natural Language Processing Framework',
            'IoT Security Solutions'
        ],
        'Abstract': [
            'This project focuses on developing advanced machine learning algorithms for image recognition tasks.',
            'Research on scalable cloud computing infrastructure for enterprise applications.',
            'Exploring data mining techniques for extracting valuable insights from large datasets.',
            'Development of a comprehensive framework for natural language processing applications.',
            'Investigating security challenges and solutions for Internet of Things devices.'
        ],
        'Funding_Requested': [50000, 75000, 30000, 60000, 45000]
    }
    
    past_df = pd.DataFrame(past_data)
    past_df.to_csv('sample_past_proposals.csv', index=False)
    
    # Sample new proposals
    new_data = {
        'Proposal_ID': ['N001', 'N002', 'N003', 'N004', 'N005'],
        'Title': [
            'Deep Learning for Medical Imaging',
            'Blockchain for Supply Chain Management',
            'Reinforcement Learning in Robotics',
            'Computer Vision for Autonomous Vehicles',
            'Quantum Computing Algorithms'
        ],
        'Abstract': [
            'Using deep learning techniques to improve medical imaging diagnostics and accuracy.',
            'Applying blockchain technology to enhance transparency and security in supply chains.',
            'Research on reinforcement learning algorithms for adaptive robotic control systems.',
            'Developing computer vision systems for real-time decision making in autonomous vehicles.',
            'Exploring quantum computing algorithms for solving complex computational problems.'
        ],
        'Funding_Requested': [80000, 70000, 65000, 90000, 100000]
    }
    
    new_df = pd.DataFrame(new_data)
    new_df.to_csv('sample_new_proposals.csv', index=False)
    
    print("Sample data created: sample_past_proposals.csv and sample_new_proposals.csv")

if __name__ == "__main__":
    create_sample_data()