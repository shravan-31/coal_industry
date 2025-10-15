from risk_ethics_ip import RiskEthicsIPChecker, create_sample_proposal_data

def main():
    print("Demonstrating Risk, Ethics & IP Checker")
    print("=" * 50)
    
    # Create checker
    checker = RiskEthicsIPChecker()
    print("✓ Risk, Ethics & IP Checker initialized")
    
    # Create sample data
    sample_data = create_sample_proposal_data()
    print(f"✓ Sample data created with {len(sample_data)} proposals")
    
    # Evaluate a proposal
    proposal = sample_data[0]
    print(f"\nEvaluating proposal: {proposal['Title']}")
    result = checker.evaluate_proposal(proposal)
    
    print(f"Overall Risk Score: {result['overall_risk_score']:.4f}")
    print(f"Risk Level: {result['risk_level']}")
    
    # Show detailed results
    env_results = result['environmental_risks']
    if env_results['risks_identified']:
        print(f"\nEnvironmental Risks ({env_results['risk_level']}):")
        for risk in env_results['risks_identified']:
            print(f"  - {risk}")
        for rec in env_results['recommendations']:
            print(f"  → {rec}")
    else:
        print(f"\nEnvironmental Risks: None identified")
    
    safety_results = result['safety_risks']
    if safety_results['risks_identified']:
        print(f"\nSafety Risks ({safety_results['risk_level']}):")
        for risk in safety_results['risks_identified']:
            print(f"  - {risk}")
        for rec in safety_results['recommendations']:
            print(f"  → {rec}")
    else:
        print(f"\nSafety Risks: None identified")
    
    ethics_results = result['ethics_compliance']
    print(f"\nEthics Compliance ({ethics_results['compliance_level']}):")
    if ethics_results['issues_identified']:
        for issue in ethics_results['issues_identified']:
            print(f"  - {issue}")
    for rec in ethics_results['recommendations']:
        print(f"  → {rec}")
    
    print("\n✓ Risk, Ethics & IP Checker demonstration completed successfully")

if __name__ == "__main__":
    main()