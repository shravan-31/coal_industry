import unittest
import os
import sqlite3
import json
from database import ProposalDatabase

class TestProposalDatabase(unittest.TestCase):
    
    def setUp(self):
        """Set up test database before each test."""
        self.db_file = "test_proposals.db"
        self.db = ProposalDatabase(self.db_file)
    
    def tearDown(self):
        """Clean up test database after each test."""
        if os.path.exists(self.db_file):
            os.remove(self.db_file)
    
    def test_init_database(self):
        """Test database initialization."""
        # Check that database file was created
        self.assertTrue(os.path.exists(self.db_file))
        
        # Check that tables were created
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Check proposals table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='proposals'")
        self.assertIsNotNone(cursor.fetchone())
        
        # Check other important tables
        tables_to_check = [
            'proposal_sections', 'evaluation_scores', 'evaluation_details',
            'feedback', 'reviewers', 'evaluation_history',
            'model_performance', 'audit_log'
        ]
        
        for table in tables_to_check:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            self.assertIsNotNone(cursor.fetchone(), f"Table {table} not found")
        
        conn.close()
    
    def test_insert_proposal(self):
        """Test inserting a proposal."""
        proposal_data = {
            'proposal_id': 'TEST001',
            'title': 'Test Proposal',
            'abstract': 'This is a test proposal',
            'funding_requested': 50000,
            'pi_name': 'Dr. Test Researcher',
            'organization': 'Test University',
            'contact_email': 'test@university.edu',
            'sections': {
                'objectives': 'Test objectives',
                'methodology': 'Test methodology'
            }
        }
        
        proposal_id = self.db.insert_proposal(proposal_data)
        self.assertEqual(proposal_id, 'TEST001')
        
        # Verify proposal was inserted
        retrieved_proposal = self.db.get_proposal('TEST001')
        self.assertIsNotNone(retrieved_proposal)
        self.assertEqual(retrieved_proposal['title'], 'Test Proposal')
        self.assertEqual(retrieved_proposal['funding_requested'], 50000)
        self.assertIn('objectives', retrieved_proposal['sections'])
    
    def test_get_proposal_not_found(self):
        """Test getting a non-existent proposal."""
        proposal = self.db.get_proposal('NONEXISTENT')
        self.assertIsNone(proposal)
    
    def test_insert_evaluation_scores(self):
        """Test inserting evaluation scores."""
        # First insert a proposal
        proposal_data = {
            'proposal_id': 'EVAL001',
            'title': 'Evaluation Test Proposal'
        }
        self.db.insert_proposal(proposal_data)
        
        # Insert evaluation scores
        scores = {
            'novelty_score': 0.85,
            'financial_score': 0.75,
            'technical_score': 0.90,
            'overall_score': 83.5,
            'detailed_scores': {
                'novelty': {'score': 0.85, 'details': {'similar_count': 3}}
            }
        }
        
        score_id = self.db.insert_evaluation_scores('EVAL001', scores)
        self.assertIsInstance(score_id, int)
        self.assertGreater(score_id, 0)
        
        # Verify scores were inserted
        retrieved_scores = self.db.get_evaluation_scores('EVAL001')
        self.assertEqual(len(retrieved_scores), 1)
        score = retrieved_scores[0]
        self.assertEqual(score['novelty_score'], 0.85)
        self.assertEqual(score['overall_score'], 83.5)
        self.assertIn('novelty', score['detailed_scores'])
    
    def test_get_evaluation_scores_empty(self):
        """Test getting evaluation scores for non-existent proposal."""
        scores = self.db.get_evaluation_scores('NONEXISTENT')
        self.assertEqual(scores, [])
    
    def test_insert_feedback(self):
        """Test inserting feedback."""
        # First insert a proposal
        proposal_data = {
            'proposal_id': 'FEEDBACK001',
            'title': 'Feedback Test Proposal'
        }
        self.db.insert_proposal(proposal_data)
        
        # Insert feedback
        feedback_data = {
            'comments': 'Excellent proposal',
            'rating': 4.8,
            'accept': True,
            'override_scores': {'technical_score': 0.95}
        }
        
        feedback_id = self.db.insert_feedback('FEEDBACK001', 'REV001', feedback_data)
        self.assertIsInstance(feedback_id, int)
        self.assertGreater(feedback_id, 0)
        
        # Verify feedback was inserted
        retrieved_feedback = self.db.get_feedback('FEEDBACK001')
        self.assertEqual(len(retrieved_feedback), 1)
        feedback = retrieved_feedback[0]
        self.assertEqual(feedback['reviewer_id'], 'REV001')
        self.assertEqual(feedback['rating'], 4.8)
        self.assertTrue(feedback['accept'])
        self.assertIn('technical_score', json.loads(feedback['override_scores']))
    
    def test_insert_reviewer(self):
        """Test inserting a reviewer."""
        reviewer_data = {
            'reviewer_id': 'REV_TEST',
            'name': 'Dr. Test Reviewer',
            'email': 'test@reviewer.org',
            'organization': 'Review Institute',
            'expertise': 'AI, Machine Learning',
            'is_active': True
        }
        
        reviewer_id = self.db.insert_reviewer(reviewer_data)
        self.assertEqual(reviewer_id, 'REV_TEST')
        
        # Verify reviewer was inserted
        retrieved_reviewer = self.db.get_reviewer('REV_TEST')
        self.assertIsNotNone(retrieved_reviewer)
        self.assertEqual(retrieved_reviewer['name'], 'Dr. Test Reviewer')
        self.assertEqual(retrieved_reviewer['expertise'], 'AI, Machine Learning')
    
    def test_get_reviewer_not_found(self):
        """Test getting a non-existent reviewer."""
        reviewer = self.db.get_reviewer('NONEXISTENT')
        self.assertIsNone(reviewer)
    
    def test_log_action(self):
        """Test logging an action."""
        details = {'test': 'data', 'value': 123}
        log_id = self.db.log_action('test_action', 'test_user', 'TEST001', details)
        
        self.assertIsInstance(log_id, int)
        self.assertGreater(log_id, 0)
        
        # Verify log entry was created
        audit_log = self.db.get_audit_log('TEST001')
        self.assertEqual(len(audit_log), 1)
        log_entry = audit_log[0]
        self.assertEqual(log_entry['action_type'], 'test_action')
        self.assertEqual(log_entry['user_id'], 'test_user')
        self.assertEqual(json.loads(log_entry['details']), details)
    
    def test_get_database_stats(self):
        """Test getting database statistics."""
        # Initially, all counts should be zero
        initial_stats = self.db.get_database_stats()
        self.assertIsInstance(initial_stats, dict)
        self.assertIn('total_proposals', initial_stats)
        self.assertIn('total_evaluations', initial_stats)
        self.assertIn('total_feedback', initial_stats)
        self.assertIn('total_reviewers', initial_stats)
        
        # Add some data
        self.db.insert_proposal({'proposal_id': 'STAT001', 'title': 'Stats Test'})
        self.db.insert_reviewer({'reviewer_id': 'REV_STAT', 'name': 'Stats Reviewer'})
        
        # Check updated stats
        updated_stats = self.db.get_database_stats()
        self.assertEqual(updated_stats['total_proposals'], 1)
        self.assertEqual(updated_stats['total_reviewers'], 1)
    
    def test_export_to_dataframe(self):
        """Test exporting table to DataFrame."""
        # Add some data first
        self.db.insert_proposal({'proposal_id': 'DF001', 'title': 'DataFrame Test'})
        
        # Export proposals table
        df = self.db.export_to_dataframe('proposals')
        self.assertGreater(len(df), 0)
        self.assertIn('proposal_id', df.columns)
        self.assertIn('title', df.columns)
        
        # Test with empty table
        empty_df = self.db.export_to_dataframe('feedback')
        self.assertIsInstance(empty_df, type(df))
        self.assertEqual(len(empty_df), 0)

if __name__ == '__main__':
    unittest.main()