import unittest
from risk_ethics_ip import RiskEthicsIPChecker, create_sample_proposal_data

class TestRiskEthicsIPChecker(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.checker = RiskEthicsIPChecker()
        self.sample_text = "This proposal involves safety monitoring and environmental impact assessment for coal mining operations."
        self.sample_proposal = {
            'Title': 'AI for Mine Safety Monitoring',
            'Abstract': 'Using artificial intelligence to monitor and predict safety hazards in coal mines to prevent accidents. The system will detect toxic gas emissions and alert workers.',
            'Methodology': 'Deploy sensors throughout the mine to collect safety data. Use machine learning models to predict hazards.',
            'Objectives': 'Reduce mining accidents by 50% through early warning systems.',
            'Budget_Justification': 'Funding requested for sensor equipment and personnel.',
            'PI_Name': 'Dr. Jane Smith',
            'Team_Members': ['John Doe', 'Alice Johnson']
        }
    
    def test_initialization(self):
        """Test that the checker initializes correctly."""
        self.assertIsNotNone(self.checker)
        self.assertIsInstance(self.checker.environmental_risk_keywords, list)
        self.assertIsInstance(self.checker.safety_risk_keywords, list)
        self.assertIsInstance(self.checker.ethics_violation_keywords, list)
    
    def test_check_environmental_risks(self):
        """Test environmental risk checking."""
        result = self.checker.check_environmental_risks(self.sample_text)
        
        # Check that we get expected keys
        expected_keys = ['risk_score', 'risks_identified', 'risk_level', 'recommendations']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check value types
        self.assertIsInstance(result['risk_score'], float)
        self.assertIsInstance(result['risks_identified'], list)
        self.assertIsInstance(result['risk_level'], str)
        self.assertIsInstance(result['recommendations'], list)
    
    def test_check_safety_risks(self):
        """Test safety risk checking."""
        result = self.checker.check_safety_risks(self.sample_text)
        
        # Check that we get expected keys
        expected_keys = ['risk_score', 'risks_identified', 'risk_level', 'recommendations']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check value types
        self.assertIsInstance(result['risk_score'], float)
        self.assertIsInstance(result['risks_identified'], list)
        self.assertIsInstance(result['risk_level'], str)
        self.assertIsInstance(result['recommendations'], list)
    
    def test_check_ethics_compliance(self):
        """Test ethics compliance checking."""
        result = self.checker.check_ethics_compliance(self.sample_text)
        
        # Check that we get expected keys
        expected_keys = ['compliance_score', 'issues_identified', 'compliance_level', 'recommendations']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check value types
        self.assertIsInstance(result['compliance_score'], float)
        self.assertIsInstance(result['issues_identified'], list)
        self.assertIsInstance(result['compliance_level'], str)
        self.assertIsInstance(result['recommendations'], list)
    
    def test_check_conflict_of_interest(self):
        """Test conflict of interest checking."""
        result = self.checker.check_conflict_of_interest(self.sample_text, "Dr. Smith", ["John Doe"])
        
        # Check that we get expected keys
        expected_keys = ['conflict_score', 'conflicts_identified', 'conflict_level', 'recommendations']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check value types
        self.assertIsInstance(result['conflict_score'], float)
        self.assertIsInstance(result['conflicts_identified'], list)
        self.assertIsInstance(result['conflict_level'], str)
        self.assertIsInstance(result['recommendations'], list)
    
    def test_check_ip_conflicts(self):
        """Test IP conflict checking."""
        result = self.checker.check_ip_conflicts(self.sample_text)
        
        # Check that we get expected keys
        expected_keys = ['ip_conflict_score', 'conflicts_identified', 'conflict_level', 'recommendations']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check value types
        self.assertIsInstance(result['ip_conflict_score'], float)
        self.assertIsInstance(result['conflicts_identified'], list)
        self.assertIsInstance(result['conflict_level'], str)
        self.assertIsInstance(result['recommendations'], list)
    
    def test_classify_risk_level(self):
        """Test risk level classification."""
        self.assertEqual(self.checker._classify_risk_level(0.8), "High")
        self.assertEqual(self.checker._classify_risk_level(0.5), "Medium")
        self.assertEqual(self.checker._classify_risk_level(0.2), "Low")
        self.assertEqual(self.checker._classify_risk_level(0.0), "None")
    
    def test_classify_compliance_level(self):
        """Test compliance level classification."""
        self.assertEqual(self.checker._classify_compliance_level(0.9), "High")
        self.assertEqual(self.checker._classify_compliance_level(0.6), "Medium")
        self.assertEqual(self.checker._classify_compliance_level(0.3), "Low")
    
    def test_evaluate_proposal(self):
        """Test comprehensive proposal evaluation."""
        result = self.checker.evaluate_proposal(self.sample_proposal)
        
        # Check that we get expected keys
        expected_keys = [
            'overall_risk_score',
            'risk_level',
            'environmental_risks',
            'safety_risks',
            'dual_use_concerns',
            'ethics_compliance',
            'conflict_of_interest',
            'ip_conflicts',
            'restricted_technology'
        ]
        
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check value types
        self.assertIsInstance(result['overall_risk_score'], float)
        self.assertIsInstance(result['risk_level'], str)
        
        # Check nested dictionaries
        self.assertIsInstance(result['environmental_risks'], dict)
        self.assertIsInstance(result['safety_risks'], dict)
        self.assertIsInstance(result['ethics_compliance'], dict)

if __name__ == '__main__':
    unittest.main()