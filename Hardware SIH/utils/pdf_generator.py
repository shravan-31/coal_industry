import pandas as pd
from datetime import datetime

# ReportLab Imports
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

class PDFGenerator:
    """
    A class to generate professional PDF summaries for R&D proposal evaluations
    """

    def __init__(self, logo_path=None):
        """
        Initialize the PDF generator with optional logo
        """
        self.logo_path = logo_path
        self.styles = getSampleStyleSheet()
        self.custom_styles = {
            'Title': ParagraphStyle(
                'CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=18,
                spaceAfter=20,
                alignment=TA_CENTER,
                textColor=colors.darkblue
            ),
            'SectionHeader': ParagraphStyle(
                'SectionHeader',
                parent=self.styles['Heading2'],
                fontSize=14,
                spaceBefore=15,
                spaceAfter=8,
                textColor=colors.darkgreen
            ),
            'Normal': ParagraphStyle(
                'CustomNormal',
                parent=self.styles['Normal'],
                fontSize=10,
                spaceAfter=5
            ),
            'Score': ParagraphStyle(
                'Score',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.darkred,
                spaceAfter=5
            )
        }

    def _header_footer(self, canvas, doc):
        """
        Draws header and footer on every page
        """
        canvas.saveState()
        width, height = A4

        # Header: logo + title
        if self.logo_path:
            try:
                canvas.drawImage(self.logo_path, 40, height - 70, width=60, height=60, preserveAspectRatio=True)
            except:
                pass
        canvas.setFont('Helvetica-Bold', 10)
        canvas.drawCentredString(width / 2, height - 40, "NaCCER, CMPDI - Proposal Evaluation Report")

        # Footer: page number + date
        canvas.setFont('Helvetica', 8)
        canvas.drawString(40, 30, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        canvas.drawRightString(width - 40, 30, f"Page {doc.page}")

        canvas.restoreState()

    def generate_proposal_summary(self, proposal_data, output_path):
        """
        Generate a PDF summary for a single proposal evaluation
        """
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []

        # Logo & Title
        if self.logo_path:
            story.append(Image(self.logo_path, width=1*inch, height=1*inch))
        story.append(Paragraph("R&D Proposal Evaluation Summary", self.custom_styles['Title']))
        story.append(Spacer(1, 0.2*inch))

        # Basic Info
        story.append(Paragraph(f"<b>Proposal ID:</b> {proposal_data.get('Proposal_ID', 'N/A')}", self.custom_styles['Normal']))
        story.append(Paragraph(f"<b>Title:</b> {proposal_data.get('Title', 'N/A')}", self.custom_styles['Normal']))
        story.append(Spacer(1, 0.2*inch))

        # Evaluation Summary
        story.append(Paragraph("Evaluation Summary", self.custom_styles['SectionHeader']))
        story.append(Paragraph(f"<b>Overall Score:</b> {proposal_data.get('Overall_Score', 0):.2f} / 100", self.custom_styles['Score']))
        story.append(Paragraph(f"<b>Recommendation:</b> {proposal_data.get('Recommendation', 'N/A')}", self.custom_styles['Score']))
        story.append(Spacer(1, 0.2*inch))

        # Detailed Scores
        story.append(Paragraph("Detailed Scores", self.custom_styles['SectionHeader']))
        score_data = [
            ['Parameter', 'Score'],
            ['Novelty', f"{proposal_data.get('Novelty_Score', 0):.2f}"],
            ['Financial Viability', f"{proposal_data.get('Financial_Score', 0):.2f}"],
            ['Technical Feasibility', f"{proposal_data.get('Technical_Score', 0):.2f}"],
            ['Coal Sector Relevance', f"{proposal_data.get('Coal_Relevance_Score', 0):.2f}"],
            ['Government Alignment', f"{proposal_data.get('Alignment_Score', 0):.2f}"],
            ['Clarity & Structure', f"{proposal_data.get('Clarity_Score', 0):.2f}"],
            ['Socio-Economic & Environmental Impact', f"{proposal_data.get('Impact_Score', 0):.2f}"]
        ]

        score_table = Table(score_data, colWidths=[3*inch, 2*inch])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.8, colors.black)
        ]))
        story.append(score_table)
        story.append(Spacer(1, 0.3*inch))

        # Feedback Section
        if 'Feedback' in proposal_data:
            story.append(Paragraph("Evaluator Feedback", self.custom_styles['SectionHeader']))
            try:
                import ast
                feedback = ast.literal_eval(proposal_data['Feedback'])
                if feedback.get('strengths'):
                    story.append(Paragraph("<b>Strengths:</b>", self.custom_styles['Normal']))
                    for s in feedback['strengths']:
                        story.append(Paragraph(f"• {s}", self.custom_styles['Normal']))
                if feedback.get('weaknesses'):
                    story.append(Paragraph("<b>Weaknesses:</b>", self.custom_styles['Normal']))
                    for w in feedback['weaknesses']:
                        story.append(Paragraph(f"• {w}", self.custom_styles['Normal']))
                if feedback.get('suggestions'):
                    story.append(Paragraph("<b>Suggestions:</b>", self.custom_styles['Normal']))
                    for s in feedback['suggestions']:
                        story.append(Paragraph(f"• {s}", self.custom_styles['Normal']))
            except:
                story.append(Paragraph(str(proposal_data['Feedback']), self.custom_styles['Normal']))

        # Funding Info
        if 'Funding_Requested' in proposal_data:
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph("Financial Information", self.custom_styles['SectionHeader']))
            story.append(Paragraph(f"Funding Requested: ₹{proposal_data['Funding_Requested']:,.2f}", self.custom_styles['Normal']))

        # Build PDF
        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)
        return output_path

    def generate_batch_summary(self, proposals_df, output_path):
        """
        Generate a summary PDF for multiple proposals
        """
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []

        # Title
        story.append(Paragraph("R&D Proposal Evaluation Batch Summary", self.custom_styles['Title']))
        story.append(Spacer(1, 0.2*inch))

        story.append(Paragraph(f"Total Proposals Evaluated: {len(proposals_df)}", self.custom_styles['Normal']))
        avg_score = proposals_df['Overall_Score'].mean()
        story.append(Paragraph(f"Average Overall Score: {avg_score:.2f}/100", self.custom_styles['Normal']))
        story.append(Spacer(1, 0.2*inch))

        # Recommendation Distribution
        story.append(Paragraph("Recommendation Distribution", self.custom_styles['SectionHeader']))
        rec_counts = proposals_df['Recommendation'].value_counts()
        rec_data = [['Recommendation', 'Count']] + [[str(k), int(v)] for k, v in rec_counts.items()]

        rec_table = Table(rec_data, colWidths=[3*inch, 2*inch])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.8, colors.black)
        ]))
        story.append(rec_table)
        story.append(PageBreak())

        # Top Proposals
        story.append(Paragraph("Top 10 Ranked Proposals", self.custom_styles['SectionHeader']))
        top = proposals_df.nlargest(10, 'Overall_Score')

        top_data = [['Rank', 'Proposal ID', 'Title', 'Overall Score', 'Recommendation']]
        for i, (_, row) in enumerate(top.iterrows(), 1):
            title = row['Title'][:50] + '...' if len(row['Title']) > 50 else row['Title']
            top_data.append([str(i), str(row['Proposal_ID']), str(title), f"{row['Overall_Score']:.2f}", str(row['Recommendation'])])

        top_table = Table(top_data, colWidths=[0.6*inch, 1*inch, 2.8*inch, 0.9*inch, 1.3*inch])
        top_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.8, colors.black)
        ]))
        story.append(top_table)

        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)
        return output_path

def main():
    generator = PDFGenerator(logo_path="logo.png")  # Optional logo
    print("✅ PDFGenerator module is ready to generate reports.")

if __name__ == "__main__":
    main()
