import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.enhanced_evaluator_hil import EnhancedRDPEvaluatorHIL
from core.enhanced_evaluator import create_sample_data  # Import from enhanced_evaluator instead
from utils.document_parser import DocumentParser
from utils.pdf_generator import PDFGenerator

# Set page configuration
st.set_page_config(
    page_title="R&D Proposal Evaluator - HIL",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 R&D Proposal Evaluation System - Human-in-the-loop")
st.markdown("""
This enhanced system automatically evaluates R&D proposals with human-in-the-loop capabilities:
- **Dynamic Weight Adjustment**: Adjust evaluation parameter weights in real-time
- **Interactive Feedback Collection**: Collect reviewer feedback and ratings
- **Weight Retraining**: Adjust weights based on human feedback patterns
- **All previous features**: Document parsing, visualization, export capabilities
""")

# Initialize session state
if 'evaluated' not in st.session_state:
    st.session_state.evaluated = False
    st.session_state.results = None
    st.session_state.visualization_data = None
    st.session_state.weights = {
        'novelty': 0.20,
        'financial': 0.15,
        'technical': 0.15,
        'coal_relevance': 0.15,
        'alignment': 0.10,
        'clarity': 0.10,
        'impact': 0.15
    }
    st.session_state.current_reviewer_id = ""
    st.session_state.proposal_feedback = {}

# Sidebar for navigation
page = st.sidebar.selectbox("Navigation", ["Home", "Upload & Evaluate", "Dashboard", "Admin Panel", "Reviewer Feedback"])

if page == "Home":
    st.header("Welcome to the Enhanced R&D Proposal Evaluation System")
    st.markdown("""
    This AI-powered system helps evaluate R&D proposals for NaCCER (CMPDI Ranchi) by the Ministry of Coal and Coal India Limited.
    
    ### Key Features:
    - **Automated Evaluation**: AI-based scoring across 7 key parameters
    - **Document Parsing**: Extract information from PDF/DOCX files
    - **Qualitative Feedback**: Detailed strengths, weaknesses, and suggestions
    - **Visual Dashboard**: Interactive charts and analytics
    - **Export Capabilities**: Download results as CSV or PDF
    - **Human-in-the-Loop**: Dynamic weight adjustment and feedback collection
    
    ### Evaluation Parameters:
    1. Technical Feasibility
    2. Financial Viability
    3. Novelty / Innovation
    4. Relevance to Coal Sector
    5. Alignment with Government and MoC/CIL Objectives
    6. Clarity and Structure of Proposal
    7. Expected Socio-Economic and Environmental Impact
    """)
    
    # Show sample results if available
    if st.session_state.evaluated and st.session_state.results is not None:
        if st.button("View Sample Evaluation Results"):
            st.session_state.current_page = "Upload & Evaluate"

elif page == "Upload & Evaluate":
    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Option to use sample data
    use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)

    # File upload section
    st.header("Upload Proposal Files")

    # Initialize file paths and upload variables
    past_file_path = None
    new_file_path = None
    past_file = None
    new_file = None

    if use_sample_data:
        st.info("Using sample data for demonstration. Create sample files first.")
        if st.button("Create Sample Data Files"):
            create_sample_data()
            st.success("Sample data files created successfully!")
        
        past_file_path = "sample_past_proposals.csv"
        new_file_path = "sample_new_proposals.csv"
    else:
        # File uploaders
        past_file = st.file_uploader("Upload Past Proposals CSV", type=["csv"])
        new_file = st.file_uploader("Upload New Proposals CSV", type=["csv"])
        
        if past_file is not None and new_file is not None:
            # Save uploaded files temporarily
            past_file_path = "uploaded_past_proposals.csv"
            new_file_path = "uploaded_new_proposals.csv"
            
            with open(past_file_path, "wb") as f:
                f.write(past_file.getbuffer())
                
            with open(new_file_path, "wb") as f:
                f.write(new_file.getbuffer())
            
            st.success("Files uploaded successfully!")

    # Document parsing section
    st.header("Document Parsing (PDF/DOCX)")
    st.markdown("Upload individual proposals in PDF or DOCX format for parsing:")
    
    uploaded_document = st.file_uploader("Upload Proposal Document", type=["pdf", "docx"])
    
    if uploaded_document is not None:
        # Save uploaded document temporarily
        doc_path = f"temp_{uploaded_document.name}"
        with open(doc_path, "wb") as f:
            f.write(uploaded_document.getbuffer())
        
        # Parse document
        try:
            parser = DocumentParser()
            if uploaded_document.name.endswith(".pdf"):
                sections = parser.parse_pdf(doc_path)
            else:
                sections = parser.parse_docx(doc_path)
            
            st.success("Document parsed successfully!")
            
            # Display parsed sections
            st.subheader("Parsed Sections")
            for section, content in sections.items():
                if section != 'full_text' and content:
                    st.markdown(f"**{section.capitalize()}:**")
                    st.text(content[:500] + ("..." if len(content) > 500 else ""))
            
            # Option to convert to CSV format
            if st.button("Convert to Evaluation Format"):
                proposal_id = st.text_input("Proposal ID", "PROP001")
                title = st.text_input("Proposal Title", "")
                csv_data = parser.convert_to_csv_format(sections, proposal_id, title)
                
                # Display CSV data
                st.subheader("Converted Data")
                st.json(csv_data)
                
                # Option to save as CSV
                csv_content = f"{','.join(csv_data.keys())}\n{','.join(str(v) for v in csv_data.values())}"
                st.download_button(
                    label="Download as CSV",
                    data=csv_content,
                    file_name=f"{proposal_id}.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"Error parsing document: {str(e)}")
        
        # Clean up temporary file
        if os.path.exists(doc_path):
            os.remove(doc_path)

    # Evaluation section
    st.header("Evaluate Proposals")

    # Check if files are available
    files_available = bool((use_sample_data and os.path.exists("sample_past_proposals.csv") and 
                           os.path.exists("sample_new_proposals.csv")) or 
                          (not use_sample_data and past_file_path is not None and new_file_path is not None))

    if files_available:
        # Weight adjustment before evaluation
        st.subheader("Adjust Evaluation Weights")
        st.markdown("Modify the weights for each evaluation parameter before running the evaluation:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.weights['novelty'] = st.slider("Novelty Weight", 0.0, 1.0, 
                                                           st.session_state.weights['novelty'], 0.01)
            st.session_state.weights['financial'] = st.slider("Financial Weight", 0.0, 1.0, 
                                                             st.session_state.weights['financial'], 0.01)
            st.session_state.weights['technical'] = st.slider("Technical Weight", 0.0, 1.0, 
                                                             st.session_state.weights['technical'], 0.01)
            st.session_state.weights['coal_relevance'] = st.slider("Coal Relevance Weight", 0.0, 1.0, 
                                                                  st.session_state.weights['coal_relevance'], 0.01)
        
        with col2:
            st.session_state.weights['alignment'] = st.slider("Alignment Weight", 0.0, 1.0, 
                                                             st.session_state.weights['alignment'], 0.01)
            st.session_state.weights['clarity'] = st.slider("Clarity Weight", 0.0, 1.0, 
                                                           st.session_state.weights['clarity'], 0.01)
            st.session_state.weights['impact'] = st.slider("Impact Weight", 0.0, 1.0, 
                                                          st.session_state.weights['impact'], 0.01)
        
        # Validate weights sum to 1.0
        total_weight = sum(st.session_state.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"Warning: Weights sum to {total_weight:.2f}, not 1.0")
        else:
            st.success(f"Weights sum to {total_weight:.2f}")
        
        # Evaluate button
        if st.button("Evaluate Proposals with Custom Weights", type="primary"):
            try:
                with st.spinner("Evaluating proposals... This may take a moment."):
                    # Initialize evaluator
                    evaluator = EnhancedRDPEvaluatorHIL()
                    
                    # Set custom weights
                    evaluator.set_weights(st.session_state.weights)
                    
                    # Evaluate proposals
                    st.session_state.results = evaluator.evaluate_proposals(
                        past_file_path,
                        new_file_path
                    )
                    
                    # Prepare visualization data
                    st.session_state.visualization_data = st.session_state.results.copy()
                    
                    st.session_state.evaluated = True
                    st.success("Evaluation completed successfully with custom weights!")
                    
            except Exception as e:
                st.error(f"An error occurred during evaluation: {str(e)}")
    else:
        st.warning("Please upload files or use sample data to proceed.")

    # Results section
    if st.session_state.evaluated and st.session_state.results is not None:
        st.header("Evaluation Results")
        
        # Display current weights used
        st.subheader("Evaluation Weights Used")
        weights_data = list(st.session_state.weights.items())
        # Create DataFrame using dictionary approach to avoid type issues
        weights_df = pd.DataFrame({
            'Parameter': [item[0] for item in weights_data],
            'Weight': [item[1] for item in weights_data]
        })
        st.table(weights_df)
        
        # Display top proposals
        st.subheader("Top Ranked Proposals")
        top_n = st.slider("Number of top proposals to display", 1, 20, 5)
        
        # Show results table
        display_columns = [
            'Proposal_ID', 'Title', 'Novelty_Score', 'Financial_Score', 'Technical_Score', 
            'Coal_Relevance_Score', 'Alignment_Score', 'Clarity_Score', 'Impact_Score', 
            'Overall_Score', 'Recommendation'
        ]
        
        top_proposals = st.session_state.results.head(top_n)[display_columns]
        st.dataframe(top_proposals.style.format({
            'Novelty_Score': '{:.4f}',
            'Financial_Score': '{:.4f}',
            'Technical_Score': '{:.4f}',
            'Coal_Relevance_Score': '{:.4f}',
            'Alignment_Score': '{:.4f}',
            'Clarity_Score': '{:.4f}',
            'Impact_Score': '{:.4f}',
            'Overall_Score': '{:.2f}'
        }))
        
        # Download results
        st.subheader("Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = st.session_state.results.to_csv(index=False)
            st.download_button(
                label="Download Evaluated Proposals (CSV)",
                data=csv,
                file_name="evaluated_proposals.csv",
                mime="text/csv"
            )
        
        with col2:
            if st.button("Generate Batch Summary PDF"):
                try:
                    generator = PDFGenerator()
                    pdf_path = "batch_summary.pdf"
                    generator.generate_batch_summary(st.session_state.results, pdf_path)
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="Download Batch Summary (PDF)",
                            data=f,
                            file_name="batch_summary.pdf",
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
        
        # Visualization
        st.subheader("Score Distribution")
        chart_data = st.session_state.results.head(top_n)
        
        # Create a bar chart for scores
        score_columns = [
            'Novelty_Score', 'Financial_Score', 'Technical_Score', 
            'Coal_Relevance_Score', 'Alignment_Score', 'Clarity_Score', 'Impact_Score'
        ]
        
        # Filter chart data to only include selected columns and valid numeric values
        chart_data_filtered = chart_data.set_index('Proposal_ID')[score_columns]
        
        # Check if we have valid data to plot
        if not chart_data_filtered.empty and chart_data_filtered.isna().sum().sum() == 0:
            # Use numpy to check for infinite values
            chart_values = chart_data_filtered.values
            has_infinite = np.isinf(chart_values).any()
            if not has_infinite:
                st.bar_chart(chart_data_filtered)
            else:
                st.warning("Cannot display chart due to infinite values in the data.")
        else:
            st.warning("Cannot display chart due to missing or invalid data.")
        
        # Detailed view
        st.subheader("Detailed Proposal Information")
        selected_proposal = st.selectbox(
            "Select a proposal to view details:",
            st.session_state.results['Proposal_ID'].tolist()
        )
        
        if selected_proposal:
            proposal = st.session_state.results[st.session_state.results['Proposal_ID'] == selected_proposal].iloc[0]
            st.write(f"**Title:** {proposal['Title']}")
            st.write(f"**Proposal ID:** {proposal['Proposal_ID']}")
            st.write(f"**Novelty Score:** {proposal['Novelty_Score']:.4f}")
            st.write(f"**Financial Score:** {proposal['Financial_Score']:.4f}")
            st.write(f"**Technical Score:** {proposal['Technical_Score']:.4f}")
            st.write(f"**Coal Relevance Score:** {proposal['Coal_Relevance_Score']:.4f}")
            st.write(f"**Alignment Score:** {proposal['Alignment_Score']:.4f}")
            st.write(f"**Clarity Score:** {proposal['Clarity_Score']:.4f}")
            st.write(f"**Impact Score:** {proposal['Impact_Score']:.4f}")
            st.write(f"**Overall Score:** {proposal['Overall_Score']:.2f}")
            st.write(f"**Recommendation:** {proposal['Recommendation']}")
            
            # Show funding information
            if 'Funding_Requested' in proposal:
                st.write(f"**Funding Requested:** ${proposal['Funding_Requested']:,.2f}")
            
            # Show feedback if available
            if 'Feedback' in proposal:
                try:
                    import ast
                    feedback = ast.literal_eval(proposal['Feedback'])
                    st.write("**Feedback:**")
                    if feedback.get('strengths'):
                        st.write("*Strengths:*")
                        for strength in feedback['strengths']:
                            st.write(f"- {strength}")
                    if feedback.get('weaknesses'):
                        st.write("*Weaknesses:*")
                        for weakness in feedback['weaknesses']:
                            st.write(f"- {weakness}")
                    if feedback.get('suggestions'):
                        st.write("*Suggestions:*")
                        for suggestion in feedback['suggestions']:
                            st.write(f"- {suggestion}")
                except:
                    st.write(f"**Feedback:** {proposal['Feedback']}")
            
            # PDF export for individual proposal
            st.subheader("Export Proposal Summary")
            if st.button("Generate PDF Summary for this Proposal"):
                try:
                    generator = PDFGenerator()
                    pdf_path = f"proposal_{selected_proposal}_summary.pdf"
                    generator.generate_proposal_summary(proposal.to_dict(), pdf_path)
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="Download Proposal Summary (PDF)",
                            data=f,
                            file_name=f"proposal_{selected_proposal}_summary.pdf",
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")

elif page == "Dashboard":
    st.header("Analytics Dashboard")
    
    if st.session_state.evaluated and st.session_state.visualization_data is not None:
        df = st.session_state.visualization_data
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Scores Distribution", "Recommendations", "Trends"])
        
        with tab1:
            st.subheader("Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Proposals", len(df))
            with col2:
                avg_score = df['Overall_Score'].mean()
                st.metric("Average Score", f"{avg_score:.2f}")
            with col3:
                highly_recommended = len(df[df['Recommendation'] == 'Highly Recommended'])
                st.metric("Highly Recommended", highly_recommended)
            with col4:
                not_recommended = len(df[df['Recommendation'] == 'Not Recommended'])
                st.metric("Not Recommended", not_recommended)
            
            # Overall score histogram
            st.subheader("Overall Score Distribution")
            fig = px.histogram(df, x='Overall_Score', nbins=20, 
                              title="Distribution of Overall Scores",
                              labels={'Overall_Score': 'Score', 'count': 'Number of Proposals'})
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Parameter Scores Distribution")
            
            # Prepare data for radar chart
            score_columns = [
                'Novelty_Score', 'Financial_Score', 'Technical_Score', 
                'Coal_Relevance_Score', 'Alignment_Score', 'Clarity_Score', 'Impact_Score'
            ]
            
            # Average scores for radar chart
            avg_scores = [df[col].mean() for col in score_columns]
            
            # Create radar chart
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=avg_scores,
                theta=[col.replace('_Score', '') for col in score_columns],
                fill='toself',
                name='Average Scores'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=False,
                title="Average Scores Across Evaluation Parameters"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Box plot for score distributions
            st.subheader("Score Distributions by Parameter")
            score_data = []
            for col in score_columns:
                for val in df[col]:
                    score_data.append({
                        'Parameter': col.replace('_Score', ''),
                        'Score': val
                    })
            
            score_df = pd.DataFrame(score_data)
            fig2 = px.box(score_df, x='Parameter', y='Score', 
                         title="Distribution of Scores by Parameter")
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            st.subheader("Recommendation Distribution")
            
            # Count recommendations
            rec_counts = df['Recommendation'].value_counts()
            
            # Pie chart
            fig = px.pie(values=rec_counts.values, names=rec_counts.index,
                        title="Distribution of Recommendations")
            st.plotly_chart(fig, use_container_width=True)
            
            # Bar chart
            fig2 = px.bar(x=rec_counts.index, y=rec_counts.values,
                         labels={'x': 'Recommendation', 'y': 'Count'},
                         title="Recommendation Counts")
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab4:
            st.subheader("Funding vs. Score Analysis")
            
            # Scatter plot of funding vs overall score
            if 'Funding_Requested' in df.columns:
                fig = px.scatter(df, x='Funding_Requested', y='Overall_Score',
                               hover_data=['Proposal_ID', 'Title'],
                               title="Funding Requested vs. Overall Score",
                               labels={'Funding_Requested': 'Funding Requested ($)', 
                                      'Overall_Score': 'Overall Score'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("Parameter Correlations")
            corr_columns = [
                'Novelty_Score', 'Financial_Score', 'Technical_Score', 
                'Coal_Relevance_Score', 'Alignment_Score', 'Clarity_Score', 'Impact_Score',
                'Overall_Score'
            ]
            
            corr_matrix = df[corr_columns].corr()
            fig2 = px.imshow(corr_matrix, 
                            labels=dict(x="Parameters", y="Parameters", color="Correlation"),
                            x=corr_columns, y=corr_columns,
                            title="Correlation Between Evaluation Parameters",
                            color_continuous_scale='RdBu')
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Please evaluate some proposals first to see dashboard visualizations.")

elif page == "Admin Panel":
    st.header("Admin Panel")
    
    st.subheader("System Configuration")
    
    # Model selection
    st.markdown("### Text Embedding Model")
    model_type = st.selectbox("Select Model", ["sentence_bert", "tfidf"], 
                             index=0 if "sentence_bert" else 1)
    
    # Weight configuration
    st.markdown("### Evaluation Weights")
    col1, col2 = st.columns(2)
    
    with col1:
        novelty_weight = st.slider("Novelty Weight", 0.0, 1.0, 0.20, 0.01)
        financial_weight = st.slider("Financial Weight", 0.0, 1.0, 0.15, 0.01)
        technical_weight = st.slider("Technical Weight", 0.0, 1.0, 0.15, 0.01)
        coal_relevance_weight = st.slider("Coal Relevance Weight", 0.0, 1.0, 0.15, 0.01)
    
    with col2:
        alignment_weight = st.slider("Alignment Weight", 0.0, 1.0, 0.10, 0.01)
        clarity_weight = st.slider("Clarity Weight", 0.0, 1.0, 0.10, 0.01)
        impact_weight = st.slider("Impact Weight", 0.0, 1.0, 0.15, 0.01)
    
    # Validate weights sum to 1.0
    total_weight = (novelty_weight + financial_weight + technical_weight + 
                   coal_relevance_weight + alignment_weight + clarity_weight + impact_weight)
    
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"Warning: Weights sum to {total_weight:.2f}, not 1.0")
    else:
        st.success(f"Weights sum to {total_weight:.2f}")
    
    st.subheader("Data Management")
    
    # Sample data generation
    if st.button("Generate Sample Data"):
        create_sample_data()
        st.success("Sample data generated successfully!")
    
    # Clear evaluation results
    if st.button("Clear Evaluation Results"):
        st.session_state.evaluated = False
        st.session_state.results = None
        st.session_state.visualization_data = None
        st.success("Evaluation results cleared!")
    
    st.subheader("System Information")
    st.markdown("""
    **Technology Stack:**
    - Python with pandas, numpy, scikit-learn
    - Sentence-BERT for semantic analysis
    - Streamlit for web interface
    - Plotly for data visualization
    
    **Evaluation Parameters:**
    1. Novelty (20%)
    2. Financial Viability (15%)
    3. Technical Feasibility (15%)
    4. Coal Sector Relevance (15%)
    5. Government Alignment (10%)
    6. Clarity & Structure (10%)
    7. Socio-Economic & Environmental Impact (15%)
    """)

elif page == "Reviewer Feedback":
    st.header("Reviewer Feedback Collection")
    
    if not st.session_state.evaluated or st.session_state.results is None:
        st.info("Please evaluate proposals first to provide feedback.")
    else:
        st.markdown("Provide feedback on individual proposals to help improve the evaluation system.")
        
        # Reviewer identification
        st.session_state.current_reviewer_id = st.text_input("Reviewer ID", 
                                                            st.session_state.current_reviewer_id)
        
        if not st.session_state.current_reviewer_id:
            st.warning("Please enter your Reviewer ID to provide feedback.")
        else:
            # Select proposal for feedback
            selected_proposal_id = st.selectbox(
                "Select a proposal to review:",
                st.session_state.results['Proposal_ID'].tolist()
            )
            
            if selected_proposal_id:
                proposal = st.session_state.results[
                    st.session_state.results['Proposal_ID'] == selected_proposal_id].iloc[0]
                
                st.write(f"**Title:** {proposal['Title']}")
                st.write(f"**Overall Score:** {proposal['Overall_Score']:.2f}")
                st.write(f"**Recommendation:** {proposal['Recommendation']}")
                
                # Check if feedback already exists for this proposal and reviewer
                feedback_key = f"{selected_proposal_id}_{st.session_state.current_reviewer_id}"
                
                # Feedback form
                st.subheader("Provide Your Feedback")
                
                # Initialize feedback in session state if not exists
                if feedback_key not in st.session_state.proposal_feedback:
                    st.session_state.proposal_feedback[feedback_key] = {
                        "comments": "",
                        "rating": 5.0,
                        "accept": True,
                        "override_scores": {}
                    }
                
                # Get current feedback
                current_feedback = st.session_state.proposal_feedback[feedback_key]
                
                # Feedback fields
                comments = st.text_area("Comments", value=current_feedback["comments"])
                rating = st.slider("Rating (1-5)", 1.0, 5.0, current_feedback["rating"], 0.1)
                accept = st.checkbox("Accept Recommendation", value=current_feedback["accept"])
                
                # Option to override scores
                st.subheader("Score Overrides (Optional)")
                st.markdown("Override individual parameter scores if you disagree with the AI assessment:")
                
                override_scores = current_feedback["override_scores"].copy()
                
                score_columns = [
                    'Novelty_Score', 'Financial_Score', 'Technical_Score', 
                    'Coal_Relevance_Score', 'Alignment_Score', 'Clarity_Score', 'Impact_Score'
                ]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    for i, col in enumerate(score_columns):
                        if i % 2 == 0:  # Even index columns
                            current_value = override_scores.get(col.replace('_Score', ''), 
                                                              proposal[col])
                            override_scores[col.replace('_Score', '')] = st.number_input(
                                f"{col.replace('_Score', '')} Score", 
                                min_value=0.0, max_value=1.0, value=float(current_value), step=0.01
                            )
                
                with col2:
                    for i, col in enumerate(score_columns):
                        if i % 2 == 1:  # Odd index columns
                            current_value = override_scores.get(col.replace('_Score', ''), 
                                                              proposal[col])
                            override_scores[col.replace('_Score', '')] = st.number_input(
                                f"{col.replace('_Score', '')} Score", 
                                min_value=0.0, max_value=1.0, value=float(current_value), step=0.01
                            )
                
                # Update session state
                st.session_state.proposal_feedback[feedback_key] = {
                    "comments": comments,
                    "rating": rating,
                    "accept": accept,
                    "override_scores": override_scores
                }
                
                # Submit feedback button
                if st.button("Submit Feedback"):
                    if st.session_state.current_reviewer_id:
                        # In a real implementation, we would save this to a database
                        # For now, we'll just show a success message
                        st.success(f"Feedback submitted for proposal {selected_proposal_id} by reviewer {st.session_state.current_reviewer_id}!")
                        
                        # Show summary of feedback
                        st.write("**Feedback Summary:**")
                        st.write(f"Comments: {comments}")
                        st.write(f"Rating: {rating}")
                        st.write(f"Accept Recommendation: {'Yes' if accept else 'No'}")
                        
                        if override_scores:
                            st.write("**Score Overrides:**")
                            for param, score in override_scores.items():
                                st.write(f"  {param}: {score}")
                    else:
                        st.error("Please enter your Reviewer ID.")

# Information section
st.sidebar.header("About")
st.sidebar.info("""
This enhanced AI/ML system evaluates R&D proposals with human-in-the-loop capabilities:

1. **Dynamic Weight Adjustment**: Modify evaluation parameter weights in real-time
2. **Interactive Feedback Collection**: Collect reviewer feedback and ratings
3. **Semantic Similarity**: Sentence-BERT embeddings to measure novelty
4. **Multi-criteria Evaluation**: Comprehensive assessment across 7 parameters

The final score is a weighted combination of these factors, with human feedback used to improve the system.
""")


# Footer
st.markdown("---")
st.caption("Enhanced R&D Proposal Evaluation System | Powered by AI/ML with Human-in-the-Loop | NaCCER (CMPDI Ranchi)")