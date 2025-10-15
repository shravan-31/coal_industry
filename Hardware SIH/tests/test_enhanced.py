from enhanced_evaluator import EnhancedRDPEvaluator, create_sample_data
import pandas as pd

def test_enhanced_evaluator():
    """Test the enhanced evaluator with different model types"""
    # Create sample data
    create_sample_data()
    
    print("Testing Enhanced R&D Proposal Evaluator")
    print("=" * 50)
    
    # Test with Sentence-BERT
    print("\n1. Testing with Sentence-BERT model:")
    evaluator_sbert = EnhancedRDPEvaluator(model_type='sentence_bert')
    
    results_sbert = evaluator_sbert.evaluate_proposals(
        'sample_past_proposals.csv', 
        'sample_new_proposals.csv',
        include_feasibility=True
    )
    
    print("Top 3 proposals (Sentence-BERT):")
    evaluator_sbert.display_top_proposals(results_sbert, top_n=3)
    
    # Test with TF-IDF
    print("\n2. Testing with TF-IDF model:")
    evaluator_tfidf = EnhancedRDPEvaluator(model_type='tfidf')
    
    results_tfidf = evaluator_tfidf.evaluate_proposals(
        'sample_past_proposals.csv', 
        'sample_new_proposals.csv',
        include_feasibility=True
    )
    
    print("Top 3 proposals (TF-IDF):")
    evaluator_tfidf.display_top_proposals(results_tfidf, top_n=3)
    
    # Compare results
    print("\n3. Comparison of models:")
    print("Proposal ID | SBERT Score | TFIDF Score | Difference")
    print("-" * 55)
    
    for i in range(min(3, len(results_sbert))):
        sbert_score = results_sbert.iloc[i]['Evaluation_Score']
        tfidf_score = results_tfidf.iloc[i]['Evaluation_Score']
        diff = abs(sbert_score - tfidf_score)
        prop_id = results_sbert.iloc[i]['Proposal_ID']
        print(f"{prop_id:^11} | {sbert_score:^11.4f} | {tfidf_score:^11.4f} | {diff:^10.4f}")
    
    # Save results
    evaluator_sbert.save_results(results_sbert, 'enhanced_evaluated_proposals.csv')
    print("\nResults saved to 'enhanced_evaluated_proposals.csv'")

if __name__ == "__main__":
    test_enhanced_evaluator()