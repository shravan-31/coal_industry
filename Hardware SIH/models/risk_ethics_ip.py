import re
import pandas as pd
from typing import Dict, List, Any, Tuple

class RiskEthicsIPChecker:
    """
    A class to check risks, ethics, and IP conflicts in R&D proposals
    """
    
    def __init__(self):
        """
        Initialize the Risk, Ethics & IP Checker
        """
        # Environmental and safety risk keywords
        self.environmental_risk_keywords = [
            'pollution', 'contamination', 'toxic', 'hazardous', 'waste', 'emission',
            'carbon', 'greenhouse', 'climate', 'ecosystem', 'biodiversity',
            'deforestation', 'groundwater', 'soil contamination'
        ]
        
        self.safety_risk_keywords = [
            'safety', 'hazard', 'danger', 'risk', 'accident', 'injury',
            'explosion', 'fire', 'chemical', 'radiation', 'toxicity',
            'protective equipment', 'safety protocol', 'emergency'
        ]
        
        self.dual_use_keywords = [
            'dual use', 'dual-use', 'military', 'defense', 'weapon', 'munition',
            'surveillance', 'spy', 'intelligence', 'combat', 'warfare'
        ]
        
        # Ethics keywords
        self.ethics_violation_keywords = [
            'human subject', 'animal testing', 'informed consent', 'privacy',
            'confidentiality', 'data protection', 'gdpr', 'hipaa',
            'ethical approval', 'irb', 'institutional review board',
            'animal welfare', 'genetic modification', 'biotechnology'
        ]
        
        # Conflict of interest keywords
        self.conflict_interest_keywords = [
            'conflict of interest', 'financial interest', 'personal benefit',
            'stock ownership', 'board member', 'consultant', 'advisor',
            'family member', 'related party', 'vendor relationship'
        ]
        
        # IP conflict keywords
        self.ip_conflict_keywords = [
            'patent', 'trademark', 'copyright', 'intellectual property',
            'proprietary', 'license', 'licensing', 'royalty',
            'open source', 'copyleft', 'gpl', 'mit license'
        ]
        
        # Restricted/restricted keywords
        self.restricted_technology_keywords = [
            'nuclear', 'biological weapon', 'chemical weapon',
            'missile technology', 'encryption', 'cyber weapon'
        ]
    
    def check_environmental_risks(self, text: str) -> Dict[str, Any]:
        """
        Check for environmental risks in proposal text
        
        Args:
            text (str): Proposal text to analyze
            
        Returns:
            Dict[str, Any]: Environmental risk analysis results
        """
        text_lower = text.lower()
        risks_found = []
        
        for keyword in self.environmental_risk_keywords:
            if keyword in text_lower:
                risks_found.append(keyword)
        
        risk_score = min(len(risks_found) / len(self.environmental_risk_keywords), 1.0)
        
        return {
            'risk_score': risk_score,
            'risks_identified': risks_found,
            'risk_level': self._classify_risk_level(risk_score),
            'recommendations': self._generate_environmental_recommendations(risks_found)
        }
    
    def check_safety_risks(self, text: str) -> Dict[str, Any]:
        """
        Check for safety risks in proposal text
        
        Args:
            text (str): Proposal text to analyze
            
        Returns:
            Dict[str, Any]: Safety risk analysis results
        """
        text_lower = text.lower()
        risks_found = []
        
        for keyword in self.safety_risk_keywords:
            if keyword in text_lower:
                risks_found.append(keyword)
        
        risk_score = min(len(risks_found) / len(self.safety_risk_keywords), 1.0)
        
        return {
            'risk_score': risk_score,
            'risks_identified': risks_found,
            'risk_level': self._classify_risk_level(risk_score),
            'recommendations': self._generate_safety_recommendations(risks_found)
        }
    
    def check_dual_use_technology(self, text: str) -> Dict[str, Any]:
        """
        Check for dual-use technology concerns
        
        Args:
            text (str): Proposal text to analyze
            
        Returns:
            Dict[str, Any]: Dual-use technology analysis results
        """
        text_lower = text.lower()
        concerns_found = []
        
        for keyword in self.dual_use_keywords:
            if keyword in text_lower:
                concerns_found.append(keyword)
        
        concern_score = min(len(concerns_found) / len(self.dual_use_keywords), 1.0)
        
        return {
            'concern_score': concern_score,
            'concerns_identified': concerns_found,
            'risk_level': self._classify_risk_level(concern_score),
            'recommendations': self._generate_dual_use_recommendations(concerns_found)
        }
    
    def check_ethics_compliance(self, text: str) -> Dict[str, Any]:
        """
        Check for ethics compliance issues
        
        Args:
            text (str): Proposal text to analyze
            
        Returns:
            Dict[str, Any]: Ethics compliance analysis results
        """
        text_lower = text.lower()
        issues_found = []
        
        for keyword in self.ethics_violation_keywords:
            if keyword in text_lower:
                issues_found.append(keyword)
        
        # If ethics-related keywords are found, it might indicate good compliance
        # If none are found, it might indicate lack of ethics consideration
        if len(issues_found) > 0:
            compliance_score = min(len(issues_found) / len(self.ethics_violation_keywords), 1.0)
        else:
            # No ethics keywords found - potential issue
            compliance_score = 0.3  # Low score indicates need for review
        
        return {
            'compliance_score': compliance_score,
            'issues_identified': issues_found,
            'compliance_level': self._classify_compliance_level(compliance_score),
            'recommendations': self._generate_ethics_recommendations(issues_found)
        }
    
    def check_conflict_of_interest(self, text: str, pi_name: str = "", team_members: List[str] = []) -> Dict[str, Any]:
        """
        Check for conflict of interest
        
        Args:
            text (str): Proposal text to analyze
            pi_name (str): Principal investigator name
            team_members (List[str]): List of team member names
            
        Returns:
            Dict[str, Any]: Conflict of interest analysis results
        """
        text_lower = text.lower()
        conflicts_found = []
        
        for keyword in self.conflict_interest_keywords:
            if keyword in text_lower:
                conflicts_found.append(keyword)
        
        # Check for PI name in conflict contexts
        if pi_name:
            pi_patterns = [
                f"{pi_name.lower()} interest",
                f"interest {pi_name.lower()}",
                f"{pi_name.lower()} financial",
                f"financial {pi_name.lower()}"
            ]
            
            for pattern in pi_patterns:
                if pattern in text_lower:
                    conflicts_found.append(f"PI-related conflict: {pattern}")
        
        conflict_score = min(len(conflicts_found) / max(len(self.conflict_interest_keywords), 1), 1.0)
        
        return {
            'conflict_score': conflict_score,
            'conflicts_identified': conflicts_found,
            'conflict_level': self._classify_conflict_level(conflict_score),
            'recommendations': self._generate_conflict_recommendations(conflicts_found)
        }
    
    def check_ip_conflicts(self, text: str) -> Dict[str, Any]:
        """
        Check for intellectual property conflicts
        
        Args:
            text (str): Proposal text to analyze
            
        Returns:
            Dict[str, Any]: IP conflict analysis results
        """
        text_lower = text.lower()
        conflicts_found = []
        
        for keyword in self.ip_conflict_keywords:
            if keyword in text_lower:
                conflicts_found.append(keyword)
        
        conflict_score = min(len(conflicts_found) / len(self.ip_conflict_keywords), 1.0)
        
        return {
            'ip_conflict_score': conflict_score,
            'conflicts_identified': conflicts_found,
            'conflict_level': self._classify_conflict_level(conflict_score),
            'recommendations': self._generate_ip_recommendations(conflicts_found)
        }
    
    def check_restricted_technology(self, text: str) -> Dict[str, Any]:
        """
        Check for restricted technology concerns
        
        Args:
            text (str): Proposal text to analyze
            
        Returns:
            Dict[str, Any]: Restricted technology analysis results
        """
        text_lower = text.lower()
        concerns_found = []
        
        for keyword in self.restricted_technology_keywords:
            if keyword in text_lower:
                concerns_found.append(keyword)
        
        concern_score = min(len(concerns_found) / len(self.restricted_technology_keywords), 1.0)
        
        return {
            'restricted_score': concern_score,
            'concerns_identified': concerns_found,
            'risk_level': self._classify_risk_level(concern_score),
            'recommendations': self._generate_restricted_recommendations(concerns_found)
        }
    
    def _classify_risk_level(self, score: float) -> str:
        """
        Classify risk level based on score
        
        Args:
            score (float): Risk score (0-1)
            
        Returns:
            str: Risk level classification
        """
        if score >= 0.7:
            return "High"
        elif score >= 0.4:
            return "Medium"
        elif score > 0:
            return "Low"
        else:
            return "None"
    
    def _classify_compliance_level(self, score: float) -> str:
        """
        Classify compliance level based on score
        
        Args:
            score (float): Compliance score (0-1)
            
        Returns:
            str: Compliance level classification
        """
        if score >= 0.8:
            return "High"
        elif score >= 0.5:
            return "Medium"
        else:
            return "Low"
    
    def _classify_conflict_level(self, score: float) -> str:
        """
        Classify conflict level based on score
        
        Args:
            score (float): Conflict score (0-1)
            
        Returns:
            str: Conflict level classification
        """
        if score >= 0.6:
            return "High"
        elif score >= 0.3:
            return "Medium"
        else:
            return "Low"
    
    def _generate_environmental_recommendations(self, risks_found: List[str]) -> List[str]:
        """
        Generate recommendations for environmental risks
        
        Args:
            risks_found (List[str]): Environmental risks identified
            
        Returns:
            List[str]: Recommendations
        """
        recommendations = []
        
        if risks_found:
            recommendations.append("Conduct detailed environmental impact assessment")
            recommendations.append("Develop mitigation strategies for identified risks")
            recommendations.append("Obtain necessary environmental clearances")
        else:
            recommendations.append("No significant environmental risks identified")
            
        return recommendations
    
    def _generate_safety_recommendations(self, risks_found: List[str]) -> List[str]:
        """
        Generate recommendations for safety risks
        
        Args:
            risks_found (List[str]): Safety risks identified
            
        Returns:
            List[str]: Recommendations
        """
        recommendations = []
        
        if risks_found:
            recommendations.append("Develop comprehensive safety protocols")
            recommendations.append("Ensure proper protective equipment is available")
            recommendations.append("Establish emergency response procedures")
            recommendations.append("Conduct safety training for all personnel")
        else:
            recommendations.append("No significant safety risks identified")
            
        return recommendations
    
    def _generate_dual_use_recommendations(self, concerns_found: List[str]) -> List[str]:
        """
        Generate recommendations for dual-use technology concerns
        
        Args:
            concerns_found (List[str]): Dual-use concerns identified
            
        Returns:
            List[str]: Recommendations
        """
        recommendations = []
        
        if concerns_found:
            recommendations.append("Review export control regulations")
            recommendations.append("Consult with technology control officers")
            recommendations.append("Implement proper access controls")
            recommendations.append("Document all technology use cases")
        else:
            recommendations.append("No dual-use technology concerns identified")
            
        return recommendations
    
    def _generate_ethics_recommendations(self, issues_found: List[str]) -> List[str]:
        """
        Generate recommendations for ethics compliance
        
        Args:
            issues_found (List[str]): Ethics issues identified
            
        Returns:
            List[str]: Recommendations
        """
        recommendations = []
        
        if len(issues_found) > 0:
            recommendations.append("Ensure all human subject research has IRB approval")
            recommendations.append("Obtain informed consent from all participants")
            recommendations.append("Implement data protection measures")
            recommendations.append("Follow institutional ethics guidelines")
        else:
            recommendations.append("Consider including ethics compliance statement")
            recommendations.append("Ensure proper review board approvals are obtained")
            
        return recommendations
    
    def _generate_conflict_recommendations(self, conflicts_found: List[str]) -> List[str]:
        """
        Generate recommendations for conflict of interest
        
        Args:
            conflicts_found (List[str]): Conflicts identified
            
        Returns:
            List[str]: Recommendations
        """
        recommendations = []
        
        if conflicts_found:
            recommendations.append("Disclose all potential conflicts of interest")
            recommendations.append("Implement management plans for identified conflicts")
            recommendations.append("Obtain independent review of conflicted areas")
            recommendations.append("Ensure transparent reporting throughout the project")
        else:
            recommendations.append("No explicit conflicts of interest identified")
            recommendations.append("Continue monitoring for potential conflicts")
            
        return recommendations
    
    def _generate_ip_recommendations(self, conflicts_found: List[str]) -> List[str]:
        """
        Generate recommendations for IP conflicts
        
        Args:
            conflicts_found (List[str]): IP conflicts identified
            
        Returns:
            List[str]: Recommendations
        """
        recommendations = []
        
        if conflicts_found:
            recommendations.append("Conduct freedom-to-operate analysis")
            recommendations.append("Review existing patents and IP rights")
            recommendations.append("Establish IP ownership agreements")
            recommendations.append("Implement IP protection measures")
        else:
            recommendations.append("No explicit IP conflicts identified")
            recommendations.append("Consider IP protection for innovations")
            
        return recommendations
    
    def _generate_restricted_recommendations(self, concerns_found: List[str]) -> List[str]:
        """
        Generate recommendations for restricted technology
        
        Args:
            concerns_found (List[str]): Restricted technology concerns identified
            
        Returns:
            List[str]: Recommendations
        """
        recommendations = []
        
        if concerns_found:
            recommendations.append("Consult with export control compliance office")
            recommendations.append("Review ITAR and EAR regulations")
            recommendations.append("Obtain necessary licenses and approvals")
            recommendations.append("Implement proper access and security controls")
        else:
            recommendations.append("No restricted technology concerns identified")
            
        return recommendations
    
    def evaluate_proposal(self, proposal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a proposal for risks, ethics, and IP conflicts
        
        Args:
            proposal_data (Dict[str, Any]): Proposal data
            
        Returns:
            Dict[str, Any]: Comprehensive risk, ethics, and IP evaluation
        """
        # Combine all text fields for analysis
        text_fields = [
            proposal_data.get('Title', ''),
            proposal_data.get('Abstract', ''),
            proposal_data.get('Methodology', ''),
            proposal_data.get('Objectives', ''),
            proposal_data.get('Budget_Justification', ''),
            proposal_data.get('Personnel', '')
        ]
        
        full_text = ' '.join(str(field) for field in text_fields if field)
        
        # Get PI name and team members
        pi_name = proposal_data.get('PI_Name', '')
        team_members = proposal_data.get('Team_Members', [])
        
        # Perform all checks
        environmental_results = self.check_environmental_risks(full_text)
        safety_results = self.check_safety_risks(full_text)
        dual_use_results = self.check_dual_use_technology(full_text)
        ethics_results = self.check_ethics_compliance(full_text)
        conflict_results = self.check_conflict_of_interest(full_text, pi_name, team_members)
        ip_results = self.check_ip_conflicts(full_text)
        restricted_results = self.check_restricted_technology(full_text)
        
        # Calculate overall risk score (weighted average)
        risk_score = (
            0.2 * environmental_results['risk_score'] +
            0.2 * safety_results['risk_score'] +
            0.15 * dual_use_results['concern_score'] +
            0.15 * restricted_results['restricted_score'] +
            0.1 * (1 - ethics_results['compliance_score']) +  # Invert ethics score
            0.1 * conflict_results['conflict_score'] +
            0.1 * ip_results['ip_conflict_score']
        )
        
        # Compile all results
        evaluation_results = {
            'overall_risk_score': risk_score,
            'risk_level': self._classify_risk_level(risk_score),
            'environmental_risks': environmental_results,
            'safety_risks': safety_results,
            'dual_use_concerns': dual_use_results,
            'ethics_compliance': ethics_results,
            'conflict_of_interest': conflict_results,
            'ip_conflicts': ip_results,
            'restricted_technology': restricted_results
        }
        
        return evaluation_results

def create_sample_proposal_data() -> List[Dict[str, Any]]:
    """
    Create sample proposal data for demonstration
    
    Returns:
        List[Dict[str, Any]]: Sample proposal data
    """
    return [
        {
            'Proposal_ID': 'PROP001',
            'Title': 'AI for Mine Safety Monitoring',
            'Abstract': 'Using artificial intelligence to monitor and predict safety hazards in coal mines to prevent accidents. The system will detect toxic gas emissions and alert workers.',
            'Methodology': 'Deploy sensors throughout the mine to collect safety data. Use machine learning models to predict hazards.',
            'Objectives': 'Reduce mining accidents by 50% through early warning systems.',
            'Budget_Justification': 'Funding requested for sensor equipment and personnel.',
            'PI_Name': 'Dr. Jane Smith',
            'Team_Members': ['John Doe', 'Alice Johnson']
        },
        {
            'Proposal_ID': 'PROP002',
            'Title': 'Coal Quality Prediction Using Machine Learning',
            'Abstract': 'Developing machine learning models to predict coal quality based on geological and chemical data. No safety or environmental risks involved.',
            'Methodology': 'Analysis of existing coal samples using standard laboratory techniques.',
            'Objectives': 'Improve coal processing efficiency through better quality prediction.',
            'Budget_Justification': 'Funding for data analysis software and researcher time.',
            'PI_Name': 'Dr. Robert Brown',
            'Team_Members': ['Emily White', 'Michael Green']
        }
    ]

def main():
    """
    Main function to demonstrate the Risk, Ethics & IP Checker
    """
    print("Demonstrating Risk, Ethics & IP Checker")
    print("=" * 50)
    
    # Create checker
    checker = RiskEthicsIPChecker()
    print("✓ Risk, Ethics & IP Checker initialized")
    
    # Create sample data
    sample_data = create_sample_proposal_data()
    print(f"✓ Sample data created with {len(sample_data)} proposals")
    
    # Evaluate proposals
    print("\nEvaluating proposals:")
    for i, proposal in enumerate(sample_data, 1):
        print(f"\nProposal {i}: {proposal['Title']}")
        result = checker.evaluate_proposal(proposal)
        
        print(f"Overall Risk Score: {result['overall_risk_score']:.4f}")
        print(f"Risk Level: {result['risk_level']}")
        
        # Show key findings
        env_risks = result['environmental_risks']['risks_identified']
        if env_risks:
            print(f"Environmental Risks: {', '.join(env_risks)}")
        
        safety_risks = result['safety_risks']['risks_identified']
        if safety_risks:
            print(f"Safety Risks: {', '.join(safety_risks)}")
        
        ethics_issues = result['ethics_compliance']['issues_identified']
        if ethics_issues:
            print(f"Ethics Issues: {', '.join(ethics_issues)}")
    
    print("\n✓ Risk, Ethics & IP Checker demonstration completed successfully")

if __name__ == "__main__":
    main()