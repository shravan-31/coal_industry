import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import sqlite3
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

class ProposalDatabase:
    """
    A class to manage the database for R&D proposal evaluation system
    """
    
    def __init__(self, db_path: str = "proposals.db"):
        """
        Initialize the database connection
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self) -> None:
        """
        Initialize the database schema
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create proposals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS proposals (
                proposal_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                abstract TEXT,
                funding_requested REAL,
                submission_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'submitted',
                pi_name TEXT,
                organization TEXT,
                contact_email TEXT
            )
        ''')
        
        # Create proposal_sections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS proposal_sections (
                section_id INTEGER PRIMARY KEY AUTOINCREMENT,
                proposal_id TEXT,
                section_type TEXT,
                content TEXT,
                FOREIGN KEY (proposal_id) REFERENCES proposals (proposal_id)
            )
        ''')
        
        # Create evaluation_scores table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_scores (
                score_id INTEGER PRIMARY KEY AUTOINCREMENT,
                proposal_id TEXT,
                evaluator_version TEXT,
                novelty_score REAL,
                financial_score REAL,
                technical_score REAL,
                coal_relevance_score REAL,
                alignment_score REAL,
                clarity_score REAL,
                impact_score REAL,
                overall_score REAL,
                evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (proposal_id) REFERENCES proposals (proposal_id)
            )
        ''')
        
        # Create evaluation_details table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_details (
                detail_id INTEGER PRIMARY KEY AUTOINCREMENT,
                score_id INTEGER,
                parameter_name TEXT,
                parameter_score REAL,
                parameter_details TEXT,  -- JSON string for detailed analysis
                FOREIGN KEY (score_id) REFERENCES evaluation_scores (score_id)
            )
        ''')
        
        # Create feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                proposal_id TEXT,
                reviewer_id TEXT,
                comments TEXT,
                rating REAL,
                accept BOOLEAN,
                feedback_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                override_scores TEXT,  -- JSON string for score overrides
                FOREIGN KEY (proposal_id) REFERENCES proposals (proposal_id)
            )
        ''')
        
        # Create reviewers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reviewers (
                reviewer_id TEXT PRIMARY KEY,
                name TEXT,
                email TEXT,
                organization TEXT,
                expertise TEXT,
                is_active BOOLEAN DEFAULT TRUE
            )
        ''')
        
        # Create evaluation_history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                proposal_id TEXT,
                model_version TEXT,
                evaluation_data TEXT,  -- JSON string with full evaluation data
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (proposal_id) REFERENCES proposals (proposal_id)
            )
        ''')
        
        # Create model_performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT,
                metric_name TEXT,
                metric_value REAL,
                evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create audit_log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT,
                user_id TEXT,
                proposal_id TEXT,
                details TEXT,  -- JSON string with action details
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_proposal(self, proposal_data: Dict[str, Any]) -> str:
        """
        Insert a new proposal into the database
        
        Args:
            proposal_data (Dict[str, Any]): Proposal data
            
        Returns:
            str: Proposal ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        proposal_id = proposal_data.get('proposal_id', f"PROP_{int(datetime.now().timestamp())}")
        
        cursor.execute('''
            INSERT INTO proposals (
                proposal_id, title, abstract, funding_requested, 
                pi_name, organization, contact_email
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            proposal_id,
            proposal_data.get('title', ''),
            proposal_data.get('abstract', ''),
            proposal_data.get('funding_requested', 0.0),
            proposal_data.get('pi_name', ''),
            proposal_data.get('organization', ''),
            proposal_data.get('contact_email', '')
        ))
        
        # Insert sections if provided
        sections = proposal_data.get('sections', {})
        for section_type, content in sections.items():
            cursor.execute('''
                INSERT INTO proposal_sections (proposal_id, section_type, content)
                VALUES (?, ?, ?)
            ''', (proposal_id, section_type, content))
        
        conn.commit()
        conn.close()
        
        # Log the action
        self.log_action('insert_proposal', 'system', proposal_id, {'title': proposal_data.get('title', '')})
        
        return proposal_id
    
    def get_proposal(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a proposal by ID
        
        Args:
            proposal_id (str): Proposal ID
            
        Returns:
            Optional[Dict[str, Any]]: Proposal data or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get proposal data
        cursor.execute('SELECT * FROM proposals WHERE proposal_id = ?', (proposal_id,))
        proposal_row = cursor.fetchone()
        
        if not proposal_row:
            conn.close()
            return None
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        proposal_data = dict(zip(column_names, proposal_row))
        
        # Get sections
        cursor.execute('SELECT section_type, content FROM proposal_sections WHERE proposal_id = ?', (proposal_id,))
        sections = {row[0]: row[1] for row in cursor.fetchall()}
        proposal_data['sections'] = sections
        
        conn.close()
        return proposal_data
    
    def insert_evaluation_scores(self, proposal_id: str, scores: Dict[str, Any], 
                               evaluator_version: str = "1.0") -> int:
        """
        Insert evaluation scores for a proposal
        
        Args:
            proposal_id (str): Proposal ID
            scores (Dict[str, Any]): Evaluation scores
            evaluator_version (str): Version of the evaluator
            
        Returns:
            int: Score ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO evaluation_scores (
                proposal_id, evaluator_version,
                novelty_score, financial_score, technical_score,
                coal_relevance_score, alignment_score, clarity_score,
                impact_score, overall_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            proposal_id, evaluator_version,
            scores.get('novelty_score', 0.0),
            scores.get('financial_score', 0.0),
            scores.get('technical_score', 0.0),
            scores.get('coal_relevance_score', 0.0),
            scores.get('alignment_score', 0.0),
            scores.get('clarity_score', 0.0),
            scores.get('impact_score', 0.0),
            scores.get('overall_score', 0.0)
        ))
        
        score_id = cursor.lastrowid
        if score_id is None:
            score_id = 0
        
        # Insert detailed scores if provided
        detailed_scores = scores.get('detailed_scores', {})
        for param_name, param_data in detailed_scores.items():
            cursor.execute('''
                INSERT INTO evaluation_details (
                    score_id, parameter_name, parameter_score, parameter_details
                ) VALUES (?, ?, ?, ?)
            ''', (
                score_id, param_name, param_data.get('score', 0.0),
                json.dumps(param_data.get('details', {}))
            ))
        
        conn.commit()
        conn.close()
        
        # Log the action
        self.log_action('insert_evaluation', 'system', proposal_id, {
            'evaluator_version': evaluator_version,
            'overall_score': scores.get('overall_score', 0.0)
        })
        
        return score_id
    
    def get_evaluation_scores(self, proposal_id: str) -> List[Dict[str, Any]]:
        """
        Get evaluation scores for a proposal
        
        Args:
            proposal_id (str): Proposal ID
            
        Returns:
            List[Dict[str, Any]]: List of evaluation scores
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM evaluation_scores 
            WHERE proposal_id = ? 
            ORDER BY evaluation_date DESC
        ''', (proposal_id,))
        
        rows = cursor.fetchall()
        if not rows:
            conn.close()
            return []
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        scores_list = []
        for row in rows:
            score_data = dict(zip(column_names, row))
            
            # Get detailed scores
            cursor.execute('''
                SELECT parameter_name, parameter_score, parameter_details 
                FROM evaluation_details 
                WHERE score_id = ?
            ''', (score_data['score_id'],))
            
            details = {}
            for detail_row in cursor.fetchall():
                details[detail_row[0]] = {
                    'score': detail_row[1],
                    'details': json.loads(detail_row[2]) if detail_row[2] else {}
                }
            
            score_data['detailed_scores'] = details
            scores_list.append(score_data)
        
        conn.close()
        return scores_list
    
    def insert_feedback(self, proposal_id: str, reviewer_id: str, 
                       feedback_data: Dict[str, Any]) -> int:
        """
        Insert feedback for a proposal
        
        Args:
            proposal_id (str): Proposal ID
            reviewer_id (str): Reviewer ID
            feedback_data (Dict[str, Any]): Feedback data
            
        Returns:
            int: Feedback ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (
                proposal_id, reviewer_id, comments, rating, accept, override_scores
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            proposal_id, reviewer_id,
            feedback_data.get('comments', ''),
            feedback_data.get('rating', 0.0),
            feedback_data.get('accept', False),
            json.dumps(feedback_data.get('override_scores', {}))
        ))
        
        feedback_id = cursor.lastrowid
        if feedback_id is None:
            feedback_id = 0
        conn.commit()
        conn.close()
        
        # Log the action
        self.log_action('insert_feedback', reviewer_id, proposal_id, {
            'rating': feedback_data.get('rating', 0.0),
            'accept': feedback_data.get('accept', False)
        })
        
        return feedback_id
    
    def get_feedback(self, proposal_id: str) -> List[Dict[str, Any]]:
        """
        Get feedback for a proposal
        
        Args:
            proposal_id (str): Proposal ID
            
        Returns:
            List[Dict[str, Any]]: List of feedback entries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM feedback 
            WHERE proposal_id = ? 
            ORDER BY feedback_date DESC
        ''', (proposal_id,))
        
        rows = cursor.fetchall()
        if not rows:
            conn.close()
            return []
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        feedback_list = []
        for row in rows:
            feedback_data = dict(zip(column_names, row))
            # Parse JSON override_scores
            if feedback_data['override_scores']:
                feedback_data['override_scores'] = json.loads(feedback_data['override_scores'])
            feedback_list.append(feedback_data)
        
        conn.close()
        return feedback_list
    
    def insert_reviewer(self, reviewer_data: Dict[str, Any]) -> str:
        """
        Insert a new reviewer
        
        Args:
            reviewer_data (Dict[str, Any]): Reviewer data
            
        Returns:
            str: Reviewer ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        reviewer_id = reviewer_data.get('reviewer_id', f"REV_{int(datetime.now().timestamp())}")
        
        cursor.execute('''
            INSERT INTO reviewers (
                reviewer_id, name, email, organization, expertise, is_active
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            reviewer_id,
            reviewer_data.get('name', ''),
            reviewer_data.get('email', ''),
            reviewer_data.get('organization', ''),
            reviewer_data.get('expertise', ''),
            reviewer_data.get('is_active', True)
        ))
        
        conn.commit()
        conn.close()
        
        # Log the action
        self.log_action('insert_reviewer', 'system', reviewer_id, {'name': reviewer_data.get('name', '')})
        
        return reviewer_id
    
    def get_reviewer(self, reviewer_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a reviewer by ID
        
        Args:
            reviewer_id (str): Reviewer ID
            
        Returns:
            Optional[Dict[str, Any]]: Reviewer data or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM reviewers WHERE reviewer_id = ?', (reviewer_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        reviewer_data = dict(zip(column_names, row))
        
        conn.close()
        return reviewer_data
    
    def log_action(self, action_type: str, user_id: str, proposal_id: Optional[str] = None, 
                  details: Optional[Dict[str, Any]] = None) -> int:
        """
        Log an action in the audit log
        
        Args:
            action_type (str): Type of action
            user_id (str): User ID
            proposal_id (Optional[str]): Proposal ID (if applicable)
            details (Optional[Dict[str, Any]]): Action details
            
        Returns:
            int: Log ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_log (action_type, user_id, proposal_id, details)
            VALUES (?, ?, ?, ?)
        ''', (
            action_type, user_id, proposal_id,
            json.dumps(details) if details else None
        ))
        
        log_id = cursor.lastrowid
        if log_id is None:
            log_id = 0
        conn.commit()
        conn.close()
        
        return log_id
    
    def get_audit_log(self, proposal_id: Optional[str] = None, 
                     limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get audit log entries
        
        Args:
            proposal_id (Optional[str]): Filter by proposal ID
            limit (int): Maximum number of entries to return
            
        Returns:
            List[Dict[str, Any]]: List of audit log entries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if proposal_id:
            cursor.execute('''
                SELECT * FROM audit_log 
                WHERE proposal_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (proposal_id, limit))
        else:
            cursor.execute('''
                SELECT * FROM audit_log 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        if not rows:
            conn.close()
            return []
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        log_entries = []
        for row in rows:
            entry_data = dict(zip(column_names, row))
            # Parse JSON details
            if entry_data['details']:
                entry_data['details'] = json.loads(entry_data['details'])
            log_entries.append(entry_data)
        
        conn.close()
        return log_entries
    
    def export_to_dataframe(self, table_name: str) -> pd.DataFrame:
        """
        Export a table to a pandas DataFrame
        
        Args:
            table_name (str): Name of the table to export
            
        Returns:
            pd.DataFrame: DataFrame with table data
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return df
    
    def get_database_stats(self) -> Dict[str, int]:
        """
        Get database statistics
        
        Returns:
            Dict[str, int]: Database statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Count proposals
        cursor.execute('SELECT COUNT(*) FROM proposals')
        stats['total_proposals'] = cursor.fetchone()[0]
        
        # Count evaluations
        cursor.execute('SELECT COUNT(*) FROM evaluation_scores')
        stats['total_evaluations'] = cursor.fetchone()[0]
        
        # Count feedback entries
        cursor.execute('SELECT COUNT(*) FROM feedback')
        stats['total_feedback'] = cursor.fetchone()[0]
        
        # Count reviewers
        cursor.execute('SELECT COUNT(*) FROM reviewers')
        stats['total_reviewers'] = cursor.fetchone()[0]
        
        # Count audit log entries
        cursor.execute('SELECT COUNT(*) FROM audit_log')
        stats['total_audit_entries'] = cursor.fetchone()[0]
        
        conn.close()
        return stats

def create_sample_database_data(db: ProposalDatabase) -> None:
    """
    Create sample data for demonstration
    
    Args:
        db (ProposalDatabase): Database instance
    """
    # Insert sample proposals
    sample_proposals = [
        {
            'proposal_id': 'PROP001',
            'title': 'AI for Mine Safety Monitoring',
            'abstract': 'Using artificial intelligence to monitor and predict safety hazards in coal mines.',
            'funding_requested': 80000,
            'pi_name': 'Dr. Jane Smith',
            'organization': 'Mining Research Institute',
            'contact_email': 'jane.smith@mri.org',
            'sections': {
                'objectives': 'Develop AI system for real-time hazard detection',
                'methodology': 'Use computer vision and sensor fusion',
                'budget': '80K over 18 months'
            }
        },
        {
            'proposal_id': 'PROP002',
            'title': 'Coal Quality Prediction Using Machine Learning',
            'abstract': 'Developing machine learning models to predict coal quality based on geological data.',
            'funding_requested': 60000,
            'pi_name': 'Dr. Robert Brown',
            'organization': 'Coal Analytics Corp',
            'contact_email': 'robert.brown@cac.com',
            'sections': {
                'objectives': 'Predict coal quality from geological surveys',
                'methodology': 'Apply ML to geological and chemical data',
                'budget': '60K over 12 months'
            }
        }
    ]
    
    for proposal in sample_proposals:
        db.insert_proposal(proposal)
    
    # Insert sample evaluations
    sample_evaluations = [
        {
            'proposal_id': 'PROP001',
            'novelty_score': 0.85,
            'financial_score': 0.75,
            'technical_score': 0.90,
            'coal_relevance_score': 0.95,
            'alignment_score': 0.88,
            'clarity_score': 0.82,
            'impact_score': 0.90,
            'overall_score': 86.5,
            'detailed_scores': {
                'novelty': {'score': 0.85, 'details': {'similar_proposals': 3}},
                'technical_feasibility': {'score': 0.90, 'details': {'team_experience': 'high'}}
            }
        },
        {
            'proposal_id': 'PROP002',
            'novelty_score': 0.75,
            'financial_score': 0.85,
            'technical_score': 0.85,
            'coal_relevance_score': 0.90,
            'alignment_score': 0.85,
            'clarity_score': 0.90,
            'impact_score': 0.80,
            'overall_score': 83.2,
            'detailed_scores': {
                'novelty': {'score': 0.75, 'details': {'similar_proposals': 5}},
                'technical_feasibility': {'score': 0.85, 'details': {'team_experience': 'medium'}}
            }
        }
    ]
    
    for evaluation in sample_evaluations:
        proposal_id = evaluation.pop('proposal_id')
        db.insert_evaluation_scores(proposal_id, evaluation)
    
    # Insert sample reviewers
    sample_reviewers = [
        {
            'reviewer_id': 'REV001',
            'name': 'Dr. Alice Johnson',
            'email': 'alice.johnson@university.edu',
            'organization': 'State University',
            'expertise': 'AI, Machine Learning',
            'is_active': True
        },
        {
            'reviewer_id': 'REV002',
            'name': 'Dr. Michael Wilson',
            'email': 'michael.wilson@research.org',
            'organization': 'National Research Center',
            'expertise': 'Coal Engineering, Safety',
            'is_active': True
        }
    ]
    
    for reviewer in sample_reviewers:
        db.insert_reviewer(reviewer)
    
    # Insert sample feedback
    sample_feedback = [
        {
            'proposal_id': 'PROP001',
            'reviewer_id': 'REV001',
            'comments': 'Excellent technical approach with strong AI methodology',
            'rating': 4.8,
            'accept': True,
            'override_scores': {'technical_score': 0.95}
        },
        {
            'proposal_id': 'PROP002',
            'reviewer_id': 'REV002',
            'comments': 'Good proposal but needs more detail on safety considerations',
            'rating': 4.2,
            'accept': True,
            'override_scores': {}
        }
    ]
    
    for feedback in sample_feedback:
        proposal_id = feedback.pop('proposal_id')
        db.insert_feedback(proposal_id, feedback['reviewer_id'], feedback)

def main():
    """
    Main function to demonstrate the Proposal Database
    """
    print("Demonstrating Proposal Database for R&D Evaluation System")
    print("=" * 58)
    
    # Initialize database
    db = ProposalDatabase("demo_proposals.db")
    print("✓ Database initialized")
    
    # Create sample data
    create_sample_database_data(db)
    print("✓ Sample data created")
    
    # Demonstrate database operations
    print("\nDatabase Operations Demo:")
    
    # 1. Get proposal
    print("\n1. Get Proposal:")
    proposal = db.get_proposal('PROP001')
    if proposal:
        print(f"   Title: {proposal['title']}")
        print(f"   PI: {proposal['pi_name']}")
        print(f"   Funding: ${proposal['funding_requested']:,.2f}")
        print(f"   Sections: {list(proposal['sections'].keys())}")
    
    # 2. Get evaluation scores
    print("\n2. Get Evaluation Scores:")
    scores = db.get_evaluation_scores('PROP001')
    if scores:
        score = scores[0]  # Most recent
        print(f"   Overall Score: {score['overall_score']}")
        print(f"   Novelty: {score['novelty_score']}")
        print(f"   Technical: {score['technical_score']}")
        print(f"   Detailed scores: {len(score['detailed_scores'])} parameters")
    
    # 3. Get feedback
    print("\n3. Get Feedback:")
    feedback_list = db.get_feedback('PROP001')
    if feedback_list:
        feedback = feedback_list[0]  # Most recent
        print(f"   Reviewer: {feedback['reviewer_id']}")
        print(f"   Rating: {feedback['rating']}")
        print(f"   Accept: {feedback['accept']}")
        print(f"   Comments: {feedback['comments'][:50]}...")
    
    # 4. Get reviewer
    print("\n4. Get Reviewer:")
    reviewer = db.get_reviewer('REV001')
    if reviewer:
        print(f"   Name: {reviewer['name']}")
        print(f"   Email: {reviewer['email']}")
        print(f"   Expertise: {reviewer['expertise']}")
        print(f"   Active: {reviewer['is_active']}")
    
    # 5. Get audit log
    print("\n5. Get Audit Log:")
    audit_log = db.get_audit_log('PROP001', limit=5)
    print(f"   Recent actions: {len(audit_log)}")
    if audit_log:
        latest_action = audit_log[0]
        print(f"   Latest: {latest_action['action_type']} by {latest_action['user_id']}")
    
    # 6. Get database stats
    print("\n6. Database Statistics:")
    stats = db.get_database_stats()
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    # 7. Export to DataFrame
    print("\n7. Export to DataFrame:")
    proposals_df = db.export_to_dataframe('proposals')
    print(f"   Proposals DataFrame shape: {proposals_df.shape}")
    
    evaluations_df = db.export_to_dataframe('evaluation_scores')
    print(f"   Evaluations DataFrame shape: {evaluations_df.shape}")
    
    print("\n" + "=" * 58)
    print("✓ Database demonstration completed successfully")
    print(f"\nNote: Demo database saved as 'demo_proposals.db'")

if __name__ == "__main__":
    main()