import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import json
import os

class FAISSNoveltyDetector:
    """
    A class to detect novelty in R&D proposals using FAISS vector database
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the FAISS Novelty Detector
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.proposals_metadata = []
        self.dimension = None
        
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into embeddings
        
        Args:
            texts (List[str]): List of texts to encode
            
        Returns:
            np.ndarray: Embeddings of the texts
        """
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.astype('float32')
    
    def build_index(self, texts: List[str], metadata: List[Dict] = None) -> None:
        """
        Build FAISS index from texts
        
        Args:
            texts (List[str]): List of texts to index
            metadata (List[Dict]): Metadata for each text
        """
        # Encode texts
        embeddings = self.encode_texts(texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Set dimension
        self.dimension = embeddings.shape[1]
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for normalized vectors = cosine similarity
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store metadata
        self.proposals_metadata = metadata if metadata else [{} for _ in texts]
    
    def save_index(self, index_path: str, metadata_path: str = None) -> None:
        """
        Save FAISS index and metadata to disk
        
        Args:
            index_path (str): Path to save the FAISS index
            metadata_path (str): Path to save the metadata (optional)
        """
        if self.index is None:
            raise ValueError("Index not built yet. Call build_index first.")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        if metadata_path:
            with open(metadata_path, 'w') as f:
                json.dump(self.proposals_metadata, f)
    
    def load_index(self, index_path: str, metadata_path: str = None) -> None:
        """
        Load FAISS index and metadata from disk
        
        Args:
            index_path (str): Path to load the FAISS index
            metadata_path (str): Path to load the metadata (optional)
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        self.dimension = self.index.d
        
        # Load metadata
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.proposals_metadata = json.load(f)
    
    def calculate_novelty_score(self, text: str, k: int = 5) -> Tuple[float, List[Dict]]:
        """
        Calculate novelty score for a new text
        
        Args:
            text (str): Text to evaluate for novelty
            k (int): Number of nearest neighbors to consider
            
        Returns:
            Tuple[float, List[Dict]]: (novelty_score, similar_proposals)
        """
        if self.index is None:
            raise ValueError("Index not built yet. Call build_index first.")
        
        # Encode text
        embedding = self.encode_texts([text])
        
        # Normalize embedding
        faiss.normalize_L2(embedding)
        
        # Search for nearest neighbors
        similarities, indices = self.index.search(embedding, k)
        
        # Calculate novelty score (1 - max similarity)
        max_similarity = float(similarities[0][0]) if similarities[0].size > 0 else 0
        novelty_score = 1.0 - max_similarity
        
        # Get similar proposals metadata
        similar_proposals = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.proposals_metadata):
                proposal_info = self.proposals_metadata[idx].copy()
                proposal_info['similarity'] = float(similarities[0][i])
                proposal_info['novelty_contribution'] = 1.0 - float(similarities[0][i])
                similar_proposals.append(proposal_info)
        
        return novelty_score, similar_proposals
    
    def find_similar_proposals(self, text: str, k: int = 5) -> List[Dict]:
        """
        Find similar proposals to a given text
        
        Args:
            text (str): Text to find similar proposals for
            k (int): Number of similar proposals to return
            
        Returns:
            List[Dict]: List of similar proposals with similarity scores
        """
        if self.index is None:
            raise ValueError("Index not built yet. Call build_index first.")
        
        # Encode text
        embedding = self.encode_texts([text])
        
        # Normalize embedding
        faiss.normalize_L2(embedding)
        
        # Search for nearest neighbors
        similarities, indices = self.index.search(embedding, k)
        
        # Get similar proposals metadata
        similar_proposals = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.proposals_metadata):
                proposal_info = self.proposals_metadata[idx].copy()
                proposal_info['similarity'] = float(similarities[0][i])
                similar_proposals.append(proposal_info)
        
        return similar_proposals

def create_proposal_kb_from_csv(csv_path: str, index_path: str, metadata_path: str) -> None:
    """
    Create a proposal knowledge base from CSV file
    
    Args:
        csv_path (str): Path to CSV file with proposals
        index_path (str): Path to save the FAISS index
        metadata_path (str): Path to save the metadata
    """
    # Load proposals
    df = pd.read_csv(csv_path)
    
    # Combine title and abstract for embedding
    texts = []
    metadata = []
    
    for _, row in df.iterrows():
        text = f"{row['Title']} {row['Abstract']}"
        texts.append(text)
        
        # Store metadata
        meta = {
            'proposal_id': row['Proposal_ID'],
            'title': row['Title'],
            'funding_requested': float(row['Funding_Requested']) if 'Funding_Requested' in row else 0.0
        }
        
        # Add any other available fields
        for col in df.columns:
            if col not in ['Proposal_ID', 'Title', 'Abstract', 'Funding_Requested']:
                meta[col.lower()] = row[col]
                
        metadata.append(meta)
    
    # Create detector and build index
    detector = FAISSNoveltyDetector()
    detector.build_index(texts, metadata)
    detector.save_index(index_path, metadata_path)
    
    print(f"Knowledge base created with {len(texts)} proposals")
    print(f"Index saved to: {index_path}")
    print(f"Metadata saved to: {metadata_path}")

def main():
    """
    Main function to demonstrate the FAISS Novelty Detector
    """
    # Create sample data if it doesn't exist
    if not os.path.exists('sample_past_proposals.csv'):
        from enhanced_evaluator import create_sample_data
        create_sample_data()
    
    # Create knowledge base
    print("Creating proposal knowledge base...")
    create_proposal_kb_from_csv(
        'sample_past_proposals.csv', 
        'proposals.index', 
        'proposals_metadata.json'
    )
    
    # Load the knowledge base
    detector = FAISSNoveltyDetector()
    detector.load_index('proposals.index', 'proposals_metadata.json')
    
    # Test with a new proposal
    new_proposal = "Using quantum computing algorithms to optimize coal extraction processes"
    
    print(f"\nEvaluating novelty for: {new_proposal}")
    
    # Calculate novelty score
    novelty_score, similar_proposals = detector.calculate_novelty_score(new_proposal, k=3)
    
    print(f"Novelty Score: {novelty_score:.4f}")
    print("\nTop 3 similar proposals:")
    for i, proposal in enumerate(similar_proposals, 1):
        print(f"{i}. {proposal['title']} (ID: {proposal['proposal_id']}) - Similarity: {proposal['similarity']:.4f}")

if __name__ == "__main__":
    main()