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

class EnhancedRDPEvaluator:
    def __init__(self, model_type='sentence_bert'):
        """
        Initialize the Enhanced R&D Proposal Evaluator
        
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
        self.novelty_weight = 0.20
        self.financial_weight = 0.15
        self.technical_weight = 0.15
        self.coal_relevance_weight = 0.15
        self.alignment_weight = 0.10
        self.clarity_weight = 0.10
        self.impact_weight = 0.15
        
        # Technical keywords for feasibility scoring
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
        
        # Government and MoC/CIL alignment keywords
        self.alignment_keywords = [
            'government', 'policy', 'regulation', 'compliance', 'standards',
            'ministry', 'coal', 'cil', 'naccer', 'cmpdi', 'ranchi',
            'sustainable', 'clean energy', 'carbon', 'emission',
            'efficiency', 'productivity', 'safety', 'health',
            'economic growth', 'employment', 'innovation', 'technology',
            'infrastructure', 'development', 'modernization'
        ]
        
        # Clarity and structure keywords
        self.clarity_indicators = [
            'objective', 'goal', 'aim', 'method', 'approach', 'technique',
            'procedure', 'process', 'step', 'phase', 'stage', 'plan',
            'budget', 'cost', 'timeline', 'schedule', 'deliverable',
            'outcome', 'result', 'benefit', 'impact', 'conclusion'
        ]
        
        # Environmental and socio-economic impact keywords
        self.impact_keywords = [
            'environment', 'pollution', 'emission', 'waste', 'recycling',
            'sustainability', 'green', 'clean', 'renewable', 'carbon footprint',
            'community', 'social', 'economic', 'employment', 'job',
            'local', 'indigenous', 'health', 'safety', 'wellbeing',
            'benefit', 'improvement', 'enhancement', 'development',
            'rehabilitation', 'restoration', 'conservation'
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
        else:
            # Default case - return zero similarity
            similarities = np.array([[0.0]])
        
        # Novelty is inverse of maximum similarity (higher similarity = less novel)
        max_similarity = np.max(similarities)
        novelty_score = 1 - max_similarity
        
        return novelty_score
    
    def calculate_financial_score(self, funding_requested, past_fundings):
        """
        Calculate normalized financial viability score (lower funding = higher score)
        
        Args:
            funding_requested (float): Funding requested for new proposal
            past_fundings (list): List of fundings from past proposals
            
        Returns:
            float: Normalized financial score
        """
        # Combine past fundings with current funding for normalization
        all_fundings = list(past_fundings) + [funding_requested]
        
        # Normalize using Min-Max scaling
        scaler = MinMaxScaler()
        normalized_fundings = scaler.fit_transform(np.array(all_fundings).reshape(-1, 1)).flatten()
        
        # Get the normalized value for the current proposal
        current_normalized = normalized_fundings[-1]
        
        # Since lower funding should get higher score, we invert the normalized value
        financial_score = 1 - current_normalized
        
        return financial_score
    
    def calculate_technical_score(self, title, abstract):
        """
        Calculate technical feasibility score based on keyword matching
        
        Args:
            title (str): Title of the proposal
            abstract (str): Abstract of the proposal
            
        Returns:
            float: Technical feasibility score
        """
        # Combine title and abstract
        full_text = (title + " " + abstract).lower()
        
        # Count matching technical keywords
        matched_keywords = 0
        for keyword in self.technical_keywords:
            if keyword.lower() in full_text:
                matched_keywords += 1
        
        # Normalize score (0 to 1)
        technical_score = matched_keywords / len(self.technical_keywords)
        
        return technical_score
    
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
            'equipment', 'machinery', 'automation', 'control', 'sensors', 'instrumentation',
            'coking', 'metallurgy', 'carbonization', 'coalbed', 'methane recovery',
            'underground', 'surface', 'opencast', 'drilling', 'blasting'
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
    
    def calculate_alignment_score(self, title, abstract):
        """
        Calculate alignment with government and MoC/CIL objectives score
        
        Args:
            title (str): Title of the proposal
            abstract (str): Abstract of the proposal
            
        Returns:
            float: Alignment score
        """
        # Combine title and abstract
        full_text = (title + " " + abstract).lower()
        
        # Count matching alignment keywords
        matched_keywords = 0
        for keyword in self.alignment_keywords:
            if keyword.lower() in full_text:
                matched_keywords += 1
        
        # Normalize score (0 to 1)
        alignment_score = matched_keywords / len(self.alignment_keywords)
        
        return alignment_score
    
    def calculate_clarity_score(self, title, abstract):
        """
        Calculate clarity and structure score based on presence of organizational keywords
        
        Args:
            title (str): Title of the proposal
            abstract (str): Abstract of the proposal
            
        Returns:
            float: Clarity and structure score
        """
        # Combine title and abstract
        full_text = (title + " " + abstract).lower()
        
        # Count matching clarity indicators
        matched_keywords = 0
        for keyword in self.clarity_indicators:
            if keyword.lower() in full_text:
                matched_keywords += 1
        
        # Normalize score (0 to 1)
        clarity_score = matched_keywords / len(self.clarity_indicators)
        
        return clarity_score
    
    def calculate_impact_score(self, title, abstract):
        """
        Calculate expected socio-economic and environmental impact score
        
        Args:
            title (str): Title of the proposal
            abstract (str): Abstract of the proposal
            
        Returns:
            float: Impact score
        """
        # Combine title and abstract
        full_text = (title + " " + abstract).lower()
        
        # Count matching impact keywords
        matched_keywords = 0
        for keyword in self.impact_keywords:
            if keyword.lower() in full_text:
                matched_keywords += 1
        
        # Normalize score (0 to 1)
        impact_score = matched_keywords / len(self.impact_keywords)
        
        return impact_score
    
    def generate_feedback(self, proposal_data, scores):
        """
        Generate qualitative feedback (strengths, weaknesses, suggestions)
        
        Args:
            proposal_data (dict): Proposal data including title and abstract
            scores (dict): Dictionary of scores for each evaluation parameter
            
        Returns:
            dict: Feedback including strengths, weaknesses, and suggestions
        """
        strengths = []
        weaknesses = []
        suggestions = []
        
        # Analyze scores and generate feedback
        if scores['novelty'] >= 0.8:
            strengths.append("Highly novel proposal with unique approach")
        elif scores['novelty'] >= 0.6:
            strengths.append("Moderately novel proposal")
        else:
            weaknesses.append("Proposal lacks novelty compared to existing work")
            suggestions.append("Consider incorporating more innovative approaches or technologies")
        
        if scores['financial'] >= 0.8:
            strengths.append("Excellent financial planning with reasonable budget")
        elif scores['financial'] >= 0.6:
            strengths.append("Good financial planning")
        else:
            weaknesses.append("Financial requirements seem high for the proposed scope")
            suggestions.append("Review budget allocation and justify costs more clearly")
        
        if scores['technical'] >= 0.8:
            strengths.append("Strong technical approach with clear methodology")
        elif scores['technical'] >= 0.6:
            strengths.append("Adequate technical approach")
        else:
            weaknesses.append("Technical approach needs more detail")
            suggestions.append("Provide more specifics on methodology and implementation")
        
        if scores['coal_relevance'] >= 0.8:
            strengths.append("Highly relevant to coal sector")
        elif scores['coal_relevance'] >= 0.6:
            strengths.append("Relevant to coal sector")
        else:
            weaknesses.append("Limited relevance to coal sector")
            suggestions.append("Better align proposal with coal industry needs and challenges")
        
        if scores['alignment'] >= 0.8:
            strengths.append("Well aligned with government and MoC/CIL objectives")
        elif scores['alignment'] >= 0.6:
            strengths.append("Moderately aligned with government objectives")
        else:
            weaknesses.append("Limited alignment with government/MoC/CIL objectives")
            suggestions.append("Better connect proposal to national priorities and coal ministry goals")
        
        if scores['clarity'] >= 0.8:
            strengths.append("Excellent clarity and structure")
        elif scores['clarity'] >= 0.6:
            strengths.append("Good clarity and structure")
        else:
            weaknesses.append("Proposal structure and clarity could be improved")
            suggestions.append("Organize content better with clear sections and objectives")
        
        if scores['impact'] >= 0.8:
            strengths.append("Significant expected socio-economic and environmental impact")
        elif scores['impact'] >= 0.6:
            strengths.append("Moderate expected impact")
        else:
            weaknesses.append("Limited discussion of expected impact")
            suggestions.append("Better articulate the socio-economic and environmental benefits")
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "suggestions": suggestions
        }
    
    def generate_recommendation(self, overall_score):
        """
        Generate AI recommendation based on overall score
        
        Args:
            overall_score (float): Overall evaluation score (0-100)
            
        Returns:
            str: Recommendation ("Highly Recommended", "Recommended with Modifications", "Not Recommended")
        """
        if overall_score >= 85:
            return "Highly Recommended"
        elif overall_score >= 70:
            return "Recommended with Modifications"
        else:
            return "Not Recommended"
    
    def evaluate_proposals(self, past_proposals_path, new_proposals_path):
        """
        Main evaluation function to score and rank proposals
        
        Args:
            past_proposals_path (str): Path to past proposals CSV
            new_proposals_path (str): Path to new proposals CSV
            
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
        financial_scores = []
        technical_scores = []
        coal_relevance_scores = []
        alignment_scores = []
        clarity_scores = []
        impact_scores = []
        overall_scores = []
        feedbacks = []
        recommendations = []
        
        # Evaluate each new proposal
        for index, proposal in new_proposals.iterrows():
            # Calculate all scores
            novelty_score = self.calculate_novelty_score(proposal['Abstract'], past_abstracts)
            financial_score = self.calculate_financial_score(proposal['Funding_Requested'], past_fundings)
            technical_score = self.calculate_technical_score(proposal['Title'], proposal['Abstract'])
            coal_relevance_score = self.calculate_coal_relevance_score(proposal['Title'], proposal['Abstract'])
            alignment_score = self.calculate_alignment_score(proposal['Title'], proposal['Abstract'])
            clarity_score = self.calculate_clarity_score(proposal['Title'], proposal['Abstract'])
            impact_score = self.calculate_impact_score(proposal['Title'], proposal['Abstract'])
            
            # Store scores
            novelty_scores.append(novelty_score)
            financial_scores.append(financial_score)
            technical_scores.append(technical_score)
            coal_relevance_scores.append(coal_relevance_score)
            alignment_scores.append(alignment_score)
            clarity_scores.append(clarity_score)
            impact_scores.append(impact_score)
            
            # Calculate overall score (0-100)
            overall_score = (
                self.novelty_weight * novelty_score +
                self.financial_weight * financial_score +
                self.technical_weight * technical_score +
                self.coal_relevance_weight * coal_relevance_score +
                self.alignment_weight * alignment_score +
                self.clarity_weight * clarity_score +
                self.impact_weight * impact_score
            ) * 100  # Scale to 0-100
            
            overall_scores.append(overall_score)
            
            # Generate feedback
            scores_dict = {
                'novelty': novelty_score,
                'financial': financial_score,
                'technical': technical_score,
                'coal_relevance': coal_relevance_score,
                'alignment': alignment_score,
                'clarity': clarity_score,
                'impact': impact_score
            }
            
            feedback = self.generate_feedback(proposal, scores_dict)
            feedbacks.append(feedback)
            
            # Generate recommendation
            recommendation = self.generate_recommendation(overall_score)
            recommendations.append(recommendation)
        
        # Add scores to the dataframe
        new_proposals['Novelty_Score'] = novelty_scores
        new_proposals['Financial_Score'] = financial_scores
        new_proposals['Technical_Score'] = technical_scores
        new_proposals['Coal_Relevance_Score'] = coal_relevance_scores
        new_proposals['Alignment_Score'] = alignment_scores
        new_proposals['Clarity_Score'] = clarity_scores
        new_proposals['Impact_Score'] = impact_scores
        new_proposals['Overall_Score'] = overall_scores
        new_proposals['Feedback'] = [str(f) for f in feedbacks]
        new_proposals['Recommendation'] = recommendations
        
        # Sort by overall score (descending)
        new_proposals = new_proposals.sort_values(by='Overall_Score', ascending=False)
        
        return new_proposals
    
    def save_results(self, evaluated_proposals, output_path):
        """
        Save evaluated proposals to CSV
        
        Args:
            evaluated_proposals (DataFrame): Evaluated proposals
            output_path (str): Path to save the output CSV
        """
        # Select columns to save
        output_columns = [
            'Proposal_ID', 'Title', 'Novelty_Score', 'Financial_Score', 
            'Technical_Score', 'Coal_Relevance_Score', 'Alignment_Score',
            'Clarity_Score', 'Impact_Score', 'Overall_Score', 'Recommendation'
        ]
            
        output_df = evaluated_proposals[output_columns]
        output_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

def main():
    """
    Main function to demonstrate the evaluator
    """
    # Initialize evaluator
    evaluator = EnhancedRDPEvaluator()
    
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
        'sample_new_proposals.csv'
    )
    
    # Save results
    evaluator.save_results(evaluated_proposals, 'enhanced_evaluated_proposals.csv')
    
    # Display results
    print("\nEvaluation Results:")
    print("=" * 100)
    for index, proposal in evaluated_proposals.iterrows():
        print(f"Proposal ID: {proposal['Proposal_ID']}")
        print(f"Title: {proposal['Title']}")
        print(f"Novelty Score: {proposal['Novelty_Score']:.4f}")
        print(f"Financial Score: {proposal['Financial_Score']:.4f}")
        print(f"Technical Score: {proposal['Technical_Score']:.4f}")
        print(f"Coal Relevance Score: {proposal['Coal_Relevance_Score']:.4f}")
        print(f"Alignment Score: {proposal['Alignment_Score']:.4f}")
        print(f"Clarity Score: {proposal['Clarity_Score']:.4f}")
        print(f"Impact Score: {proposal['Impact_Score']:.4f}")
        print(f"Overall Score: {proposal['Overall_Score']:.2f}")
        print(f"Recommendation: {proposal['Recommendation']}")
        print("-" * 100)

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
            'AI for Mine Safety Monitoring',
            'Coal Quality Prediction Using Machine Learning',
            'Automated Coal Washing Process Optimization',
            'Methane Gas Detection and Alert System',
            'Sustainable Rehabilitation of Mining Areas'
        ],
        'Abstract': [
            'Using artificial intelligence to monitor and predict safety hazards in coal mines to prevent accidents.',
            'Developing machine learning models to predict coal quality based on geological and chemical data.',
            'Implementing automated control systems to optimize the coal washing and beneficiation process.',
            'Creating a real-time methane gas detection system with automated alerts for mine safety.',
            'Developing sustainable methods for rehabilitating mining areas with native vegetation and soil restoration.'
        ],
        'Funding_Requested': [80000, 60000, 75000, 55000, 70000]
    }
    
    new_df = pd.DataFrame(new_data)
    new_df.to_csv('sample_new_proposals.csv', index=False)
    
    print("Sample data created: sample_past_proposals.csv and sample_new_proposals.csv")

if __name__ == "__main__":
    create_sample_data()