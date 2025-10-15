import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import re
from typing import Dict, List, Tuple, Any

class FinancialViabilityEvaluator:
    """
    A class to evaluate financial viability of R&D proposals using rule engine and anomaly detection
    """
    
    def __init__(self):
        """
        Initialize the Financial Viability Evaluator
        """
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.trained = False
        
        # S&T funding guidelines (example values - would be customized for MoC/CIL)
        self.guidelines = {
            'max_total_funding': 500000,  # $500K max
            'equipment_percentage_cap': 0.4,  # 40% max for equipment
            'personnel_percentage_floor': 0.5,  # At least 50% for personnel
            'overhead_percentage_cap': 0.2,  # 20% max for overhead
            'travel_percentage_cap': 0.1,  # 10% max for travel
            'indirect_costs_cap': 0.25,  # 25% max for indirect costs
        }
    
    def validate_budget_against_guidelines(self, budget_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate budget against S&T funding guidelines using rule engine
        
        Args:
            budget_data (Dict[str, float]): Budget breakdown with categories and amounts
            
        Returns:
            Dict[str, Any]: Validation results including violations and suggestions
        """
        total_funding = sum(budget_data.values())
        violations = []
        suggestions = []
        
        # Check total funding cap
        if total_funding > self.guidelines['max_total_funding']:
            violations.append({
                'rule': 'max_total_funding',
                'description': f'Total funding (${total_funding:,.2f}) exceeds maximum allowed (${self.guidelines["max_total_funding"]:,.2f})',
                'severity': 'high'
            })
            suggestions.append(f'Reduce total funding by ${total_funding - self.guidelines["max_total_funding"]:,.2f}')
        
        # Check category percentages if we have detailed breakdown
        if total_funding > 0:
            # Equipment check
            equipment_amount = budget_data.get('equipment', 0) + budget_data.get('hardware', 0)
            equipment_percentage = equipment_amount / total_funding
            if equipment_percentage > self.guidelines['equipment_percentage_cap']:
                violations.append({
                    'rule': 'equipment_percentage_cap',
                    'description': f'Equipment costs ({equipment_percentage:.1%}) exceed maximum allowed ({self.guidelines["equipment_percentage_cap"]:.0%})',
                    'severity': 'medium'
                })
                max_equipment = total_funding * self.guidelines['equipment_percentage_cap']
                suggestions.append(f'Reduce equipment costs by ${equipment_amount - max_equipment:,.2f}')
            
            # Personnel check
            personnel_amount = budget_data.get('personnel', 0) + budget_data.get('salary', 0) + budget_data.get('wages', 0)
            personnel_percentage = personnel_amount / total_funding
            if personnel_percentage < self.guidelines['personnel_percentage_floor']:
                violations.append({
                    'rule': 'personnel_percentage_floor',
                    'description': f'Personnel costs ({personnel_percentage:.1%}) below minimum required ({self.guidelines["personnel_percentage_floor"]:.0%})',
                    'severity': 'medium'
                })
                min_personnel = total_funding * self.guidelines['personnel_percentage_floor']
                suggestions.append(f'Increase personnel costs by ${min_personnel - personnel_amount:,.2f}')
            
            # Overhead check
            overhead_amount = budget_data.get('overhead', 0) + budget_data.get('indirect', 0)
            overhead_percentage = overhead_amount / total_funding
            if overhead_percentage > self.guidelines['overhead_percentage_cap']:
                violations.append({
                    'rule': 'overhead_percentage_cap',
                    'description': f'Overhead costs ({overhead_percentage:.1%}) exceed maximum allowed ({self.guidelines["overhead_percentage_cap"]:.0%})',
                    'severity': 'medium'
                })
                max_overhead = total_funding * self.guidelines['overhead_percentage_cap']
                suggestions.append(f'Reduce overhead costs by ${overhead_amount - max_overhead:,.2f}')
            
            # Travel check
            travel_amount = budget_data.get('travel', 0) + budget_data.get('transportation', 0)
            travel_percentage = travel_amount / total_funding
            if travel_percentage > self.guidelines['travel_percentage_cap']:
                violations.append({
                    'rule': 'travel_percentage_cap',
                    'description': f'Travel costs ({travel_percentage:.1%}) exceed maximum allowed ({self.guidelines["travel_percentage_cap"]:.0%})',
                    'severity': 'low'
                })
                max_travel = total_funding * self.guidelines['travel_percentage_cap']
                suggestions.append(f'Reduce travel costs by ${travel_amount - max_travel:,.2f}')
        
        # Check for disallowed items (example)
        disallowed_items = []
        for category in budget_data.keys():
            if any(disallowed in category.lower() for disallowed in ['entertainment', 'alcohol', 'personal']):
                disallowed_items.append(category)
        
        if disallowed_items:
            violations.append({
                'rule': 'disallowed_items',
                'description': f'Disallowed budget items found: {", ".join(disallowed_items)}',
                'severity': 'high'
            })
            suggestions.append(f'Remove disallowed items: {", ".join(disallowed_items)}')
        
        return {
            'valid': len([v for v in violations if v['severity'] in ['high', 'medium']]) == 0,
            'total_funding': total_funding,
            'violations': violations,
            'suggestions': suggestions
        }
    
    def train_anomaly_detector(self, historical_proposals: pd.DataFrame) -> None:
        """
        Train anomaly detection model on historical proposals
        
        Args:
            historical_proposals (pd.DataFrame): DataFrame with historical proposal data
        """
        # Extract numerical features for anomaly detection
        features = []
        for _, row in historical_proposals.iterrows():
            feature_dict = {
                'funding_requested': row.get('Funding_Requested', 0),
                'duration_months': row.get('Duration_Months', 12),  # Default 12 months
                'personnel_count': row.get('Personnel_Count', 5),  # Default 5 people
            }
            features.append(feature_dict)
        
        features_df = pd.DataFrame(features)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features_df)
        
        # Train isolation forest
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # Assume 10% of proposals are anomalies
            random_state=42
        )
        self.isolation_forest.fit(scaled_features)
        self.trained = True
    
    def detect_budget_anomalies(self, proposal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect anomalies in proposal budget using trained model
        
        Args:
            proposal_data (Dict[str, Any]): Proposal data
            
        Returns:
            Dict[str, Any]: Anomaly detection results
        """
        if not self.trained:
            raise ValueError("Anomaly detector not trained yet. Call train_anomaly_detector first.")
        
        # Extract features
        features = np.array([[
            proposal_data.get('funding_requested', 0),
            proposal_data.get('duration_months', 12),
            proposal_data.get('personnel_count', 5)
        ]])
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Predict anomalies
        anomaly_scores = self.isolation_forest.decision_function(scaled_features)
        predictions = self.isolation_forest.predict(scaled_features)
        
        # Convert to readable format
        is_anomaly = predictions[0] == -1  # Isolation Forest returns -1 for anomalies
        anomaly_score = anomaly_scores[0]
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'confidence': abs(anomaly_score)  # Higher absolute value = higher confidence
        }
    
    def calculate_expected_budget(self, proposal_features: Dict[str, Any]) -> float:
        """
        Calculate expected budget for a proposal based on its features
        
        Args:
            proposal_features (Dict[str, Any]): Features of the proposal
            
        Returns:
            float: Expected budget amount
        """
        # Simple linear regression model (would be more complex in practice)
        # Expected budget = base + (duration_factor * months) + (personnel_factor * count) + (complexity_factor * score)
        base_budget = 50000  # Base budget
        duration_factor = 2000  # $2K per month
        personnel_factor = 15000  # $15K per person
        complexity_factor = 10000  # $10K per complexity point
        
        expected_budget = (
            base_budget +
            duration_factor * proposal_features.get('duration_months', 12) +
            personnel_factor * proposal_features.get('personnel_count', 5) +
            complexity_factor * proposal_features.get('technical_complexity', 5)
        )
        
        return expected_budget
    
    def calculate_budget_zscore(self, actual_budget: float, expected_budget: float) -> float:
        """
        Calculate z-score for budget (how many standard deviations from expected)
        
        Args:
            actual_budget (float): Actual budget requested
            expected_budget (float): Expected budget based on features
            
        Returns:
            float: Z-score (positive = over budget, negative = under budget)
        """
        # In a real implementation, we would use standard deviation from historical data
        # For this example, we'll use a fixed standard deviation
        std_deviation = expected_budget * 0.3  # Assume 30% standard deviation
        
        if std_deviation == 0:
            return 0
            
        z_score = (actual_budget - expected_budget) / std_deviation
        return z_score
    
    def evaluate_proposal(self, proposal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate financial viability of a proposal
        
        Args:
            proposal_data (Dict[str, Any]): Proposal data including budget information
            
        Returns:
            Dict[str, Any]: Financial viability evaluation results
        """
        # Extract budget data
        budget_data = proposal_data.get('budget_breakdown', {})
        actual_budget = proposal_data.get('funding_requested', 0)
        
        # Validate against guidelines
        guideline_results = self.validate_budget_against_guidelines(budget_data)
        
        # Calculate expected budget and z-score
        expected_budget = self.calculate_expected_budget(proposal_data)
        z_score = self.calculate_budget_zscore(actual_budget, expected_budget)
        
        # Detect anomalies if model is trained
        anomaly_results = None
        if self.trained:
            try:
                anomaly_results = self.detect_budget_anomalies(proposal_data)
            except Exception as e:
                print(f"Warning: Could not perform anomaly detection: {e}")
        
        # Calculate financial score (0-1, higher is better)
        # Start with 1.0 and subtract penalties
        financial_score = 1.0
        
        # Penalty for guideline violations
        high_violations = len([v for v in guideline_results['violations'] if v['severity'] == 'high'])
        medium_violations = len([v for v in guideline_results['violations'] if v['severity'] == 'medium'])
        financial_score -= (high_violations * 0.2 + medium_violations * 0.1)
        
        # Penalty for extreme z-scores
        if abs(z_score) > 2:  # More than 2 standard deviations
            financial_score -= 0.2
        elif abs(z_score) > 1:  # More than 1 standard deviation
            financial_score -= 0.1
        
        # Penalty for anomalies
        if anomaly_results and anomaly_results['is_anomaly']:
            financial_score -= 0.3
        
        # Ensure score is between 0 and 1
        financial_score = max(0.0, min(1.0, financial_score))
        
        return {
            'financial_score': financial_score,
            'guideline_validation': guideline_results,
            'expected_budget': expected_budget,
            'actual_budget': actual_budget,
            'budget_zscore': z_score,
            'anomaly_detection': anomaly_results,
            'recommendations': guideline_results['suggestions']
        }

def create_sample_budget_data() -> List[Dict[str, Any]]:
    """
    Create sample budget data for demonstration
    
    Returns:
        List[Dict[str, Any]]: Sample budget data
    """
    return [
        {
            'proposal_id': 'PROP001',
            'title': 'AI for Mine Safety Monitoring',
            'funding_requested': 80000,
            'budget_breakdown': {
                'personnel': 45000,
                'equipment': 20000,
                'travel': 5000,
                'overhead': 10000
            },
            'duration_months': 18,
            'personnel_count': 3,
            'technical_complexity': 7
        },
        {
            'proposal_id': 'PROP002',
            'title': 'Coal Quality Prediction Using Machine Learning',
            'funding_requested': 60000,
            'budget_breakdown': {
                'personnel': 35000,
                'equipment': 15000,
                'travel': 3000,
                'overhead': 7000,
                'miscellaneous': 8000
            },
            'duration_months': 12,
            'personnel_count': 2,
            'technical_complexity': 6
        },
        {
            'proposal_id': 'PROP003',
            'title': 'Unrealistic Budget Proposal',
            'funding_requested': 500000,
            'budget_breakdown': {
                'equipment': 300000,  # Exceeds 40% cap
                'personnel': 50000,   # Below 50% floor
                'overhead': 150000    # Exceeds 20% cap
            },
            'duration_months': 24,
            'personnel_count': 2,
            'technical_complexity': 5
        }
    ]

def main():
    """
    Main function to demonstrate the Financial Viability Evaluator
    """
    print("Demonstrating Financial Viability Evaluator")
    print("=" * 50)
    
    # Create evaluator
    evaluator = FinancialViabilityEvaluator()
    print("✓ Financial Viability Evaluator initialized")
    
    # Create sample data
    sample_data = create_sample_budget_data()
    historical_df = pd.DataFrame(sample_data[:-1])  # Use first two as historical
    new_proposal = sample_data[-1]  # Use last as new proposal
    
    print(f"✓ Sample data created with {len(sample_data)} proposals")
    
    # Train anomaly detector
    try:
        evaluator.train_anomaly_detector(historical_df)
        print("✓ Anomaly detector trained")
    except Exception as e:
        print(f"Warning: Could not train anomaly detector: {e}")
    
    # Evaluate proposals
    print("\nEvaluating proposals:")
    for proposal in sample_data:
        print(f"\nProposal: {proposal['title']}")
        result = evaluator.evaluate_proposal(proposal)
        print(f"Financial Score: {result['financial_score']:.4f}")
        print(f"Actual Budget: ${result['actual_budget']:,.2f}")
        print(f"Expected Budget: ${result['expected_budget']:,.2f}")
        print(f"Budget Z-Score: {result['budget_zscore']:.2f}")
        
        if result['guideline_validation']['violations']:
            print("Guideline Violations:")
            for violation in result['guideline_validation']['violations']:
                print(f"  - {violation['description']} ({violation['severity']})")
        
        if result['anomaly_detection']:
            anomaly = result['anomaly_detection']
            print(f"Anomaly Detection: {'Yes' if anomaly['is_anomaly'] else 'No'} (Score: {anomaly['anomaly_score']:.2f})")
    
    print("\n✓ Financial Viability Evaluator demonstration completed successfully")

if __name__ == "__main__":
    main()