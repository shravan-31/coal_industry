import unittest
import os
import numpy as np
from novelty_detector import FAISSNoveltyDetector, create_proposal_kb_from_csv

class TestFAISSNoveltyDetector(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data
        self.sample_texts = [
            "Machine learning for image recognition in mining operations",
            "Cloud computing infrastructure for coal industry",
            "Data mining techniques for geological survey",
            "Natural language processing for safety documentation",
            "IoT security solutions for mining equipment"
        ]
        
        self.sample_metadata = [
            {"proposal_id": "P001", "title": "ML for Mining", "domain": "AI"},
            {"proposal_id": "P002", "title": "Cloud for Coal", "domain": "IT"},
            {"proposal_id": "P003", "title": "Data Mining Survey", "domain": "Data"},
            {"proposal_id": "P004", "title": "NLP Safety", "domain": "AI"},
            {"proposal_id": "P005", "title": "IoT Security", "domain": "Security"}
        ]
        
        # Create temporary files for testing
        self.index_path = "test_index.index"
        self.metadata_path = "test_metadata.json"
    
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Remove temporary files
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
    
    def test_initialization(self):
        """Test that the detector initializes correctly."""
        detector = FAISSNoveltyDetector()
        self.assertIsNotNone(detector.model)
        self.assertIsNone(detector.index)
        self.assertEqual(detector.proposals_metadata, [])
    
    def test_encode_texts(self):
        """Test text encoding functionality."""
        detector = FAISSNoveltyDetector()
        embeddings = detector.encode_texts(self.sample_texts)
        
        # Check that embeddings are numpy array
        self.assertIsInstance(embeddings, np.ndarray)
        
        # Check dimensions (should be 5 texts x 384 dimensions for all-MiniLM-L6-v2)
        self.assertEqual(embeddings.shape[0], 5)
        self.assertEqual(embeddings.shape[1], 384)
    
    def test_build_index(self):
        """Test building FAISS index."""
        detector = FAISSNoveltyDetector()
        detector.build_index(self.sample_texts, self.sample_metadata)
        
        # Check that index is built
        self.assertIsNotNone(detector.index)
        self.assertEqual(detector.index.ntotal, 5)
        self.assertEqual(len(detector.proposals_metadata), 5)
    
    def test_save_and_load_index(self):
        """Test saving and loading FAISS index."""
        # Build and save index
        detector = FAISSNoveltyDetector()
        detector.build_index(self.sample_texts, self.sample_metadata)
        detector.save_index(self.index_path, self.metadata_path)
        
        # Check that files were created
        self.assertTrue(os.path.exists(self.index_path))
        self.assertTrue(os.path.exists(self.metadata_path))
        
        # Load index
        new_detector = FAISSNoveltyDetector()
        new_detector.load_index(self.index_path, self.metadata_path)
        
        # Check that index is loaded correctly
        self.assertIsNotNone(new_detector.index)
        self.assertEqual(new_detector.index.ntotal, 5)
        self.assertEqual(len(new_detector.proposals_metadata), 5)
    
    def test_calculate_novelty_score(self):
        """Test novelty score calculation."""
        detector = FAISSNoveltyDetector()
        detector.build_index(self.sample_texts, self.sample_metadata)
        
        # Test with a similar text
        similar_text = "Advanced machine learning for mining safety"
        novelty_score, similar_proposals = detector.calculate_novelty_score(similar_text, k=3)
        
        # Check that results are valid
        self.assertIsInstance(novelty_score, float)
        self.assertTrue(0.0 <= novelty_score <= 1.0)
        self.assertIsInstance(similar_proposals, list)
        self.assertLessEqual(len(similar_proposals), 3)
        
        # Check that similar proposals have required fields
        if similar_proposals:
            proposal = similar_proposals[0]
            self.assertIn('similarity', proposal)
            self.assertIn('novelty_contribution', proposal)
    
    def test_find_similar_proposals(self):
        """Test finding similar proposals."""
        detector = FAISSNoveltyDetector()
        detector.build_index(self.sample_texts, self.sample_metadata)
        
        # Test finding similar proposals
        query_text = "Machine learning applications in mining"
        similar_proposals = detector.find_similar_proposals(query_text, k=2)
        
        # Check results
        self.assertIsInstance(similar_proposals, list)
        self.assertLessEqual(len(similar_proposals), 2)
        
        # Check that similar proposals have required fields
        if similar_proposals:
            proposal = similar_proposals[0]
            self.assertIn('proposal_id', proposal)
            self.assertIn('title', proposal)
            self.assertIn('similarity', proposal)
    
    def test_create_proposal_kb_from_csv(self):
        """Test creating knowledge base from CSV."""
        # First create sample data
        from enhanced_evaluator import create_sample_data
        create_sample_data()
        
        # Create knowledge base
        create_proposal_kb_from_csv(
            'sample_past_proposals.csv',
            self.index_path,
            self.metadata_path
        )
        
        # Check that files were created
        self.assertTrue(os.path.exists(self.index_path))
        self.assertTrue(os.path.exists(self.metadata_path))
        
        # Load and verify
        detector = FAISSNoveltyDetector()
        detector.load_index(self.index_path, self.metadata_path)
        
        self.assertIsNotNone(detector.index)
        self.assertGreater(detector.index.ntotal, 0)
        self.assertGreater(len(detector.proposals_metadata), 0)

if __name__ == '__main__':
    unittest.main()