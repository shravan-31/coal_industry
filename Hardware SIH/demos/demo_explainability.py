from explainability import ModelExplainer, MockXGBoostModel, MockLightGBMModel, create_sample_proposals
import numpy as np

def mock_predict_fn(X):
    """Mock prediction function for LIME"""
    weights = np.array([0.2, 0.15, 0.25, 0.1, 0.1, 0.05, 0.1, 0.05])
    if X.ndim == 1:
        return np.sum(X * weights)
    else:
        return np.sum(X * weights, axis=1)

def main():
    print("Demonstrating Model Explainer with SHAP and LIME")
    print("=" * 50)
    
    # Create explainer
    explainer = ModelExplainer()
    print("✓ Model Explainer initialized")
    
    # Create sample data
    sample_proposals = create_sample_proposals()
    print(f"✓ Sample data created with {len(sample_proposals)} proposals")
    
    # Prepare training data
    X_train = explainer.prepare_training_data(sample_proposals)
    print(f"✓ Training data prepared: {X_train.shape}")
    
    # Create mock models
    models = {
        'xgboost': MockXGBoostModel(),
        'lightgbm': MockLightGBMModel()
    }
    
    # Train explainers
    explainer.train_shap_explainers(models, X_train)
    explainer.train_lime_explainer(X_train)
    print("✓ SHAP and LIME explainers trained")
    
    # Explain a sample proposal
    sample_proposal = sample_proposals.iloc[0]
    X_instance = X_train[0:1]  # First proposal as instance to explain
    
    print(f"\nExplaining proposal: {sample_proposal['Title']}")
    
    # SHAP explanation
    shap_explanation = explainer.explain_with_shap('xgboost', X_instance)
    print(f"✓ SHAP explanation generated")
    
    # LIME explanation
    lime_explanation = explainer.explain_with_lime(mock_predict_fn, X_instance)
    print(f"✓ LIME explanation generated")
    
    # Feature importance
    feature_importance = explainer.get_feature_importance(models['xgboost'], X_train)
    print(f"✓ Feature importance extracted")
    
    # Generate comprehensive report
    report = explainer.generate_explanation_report(
        sample_proposal.to_dict(),
        shap_explanation,
        lime_explanation,
        feature_importance
    )
    
    # Display key findings
    print(f"\nProposal: {report['proposal_title']}")
    print(f"Overall Score: {report['overall_score']:.2f}")
    print(f"Interpretation: {report['interpretation']}")
    
    print(f"\nTop Positive Contributors (SHAP):")
    for feature, contribution in report['shap_explanation']['top_positive_contributors']:
        print(f"  {feature}: +{contribution:.4f}")
    
    print(f"\nTop Negative Contributors (SHAP):")
    for feature, contribution in report['shap_explanation']['top_negative_contributors']:
        print(f"  {feature}: {contribution:.4f}")
    
    print(f"\nFeature Importance:")
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_importance[:5]:
        print(f"  {feature}: {importance:.4f}")
    
    print("\n✓ Model Explainer demonstration completed successfully")

if __name__ == "__main__":
    main()