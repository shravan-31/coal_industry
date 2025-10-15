import PyPDF2
import re
from typing import Dict, List, Union

# Try to import docx, but handle the case where it's not available
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    # Create a dummy class for type checking purposes
    class Document:
        def __init__(self, *args, **kwargs):
            pass
    DOCX_AVAILABLE = False

class DocumentParser:
    """
    A class to parse PDF and DOCX documents and extract key sections
    """
    
    def __init__(self):
        # Define patterns for identifying sections
        self.section_patterns = {
            'abstract': [r'abstract', r'summary'],
            'objectives': [r'objective', r'goal', r'aim'],
            'methodology': [r'method', r'approach', r'technique', r'procedure'],
            'budget': [r'budget', r'cost', r'financial', r'funding'],
            'outcomes': [r'outcome', r'result', r'expected', r'deliverable']
        }
    
    def parse_pdf(self, file_path: str) -> Dict[str, str]:
        """
        Parse a PDF document and extract key sections
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            Dict[str, str]: Dictionary with extracted sections
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                return self._extract_sections(text)
        except Exception as e:
            raise Exception(f"Error parsing PDF file: {str(e)}")
    
    def parse_docx(self, file_path: str) -> Dict[str, str]:
        """
        Parse a DOCX document and extract key sections
        
        Args:
            file_path (str): Path to the DOCX file
            
        Returns:
            Dict[str, str]: Dictionary with extracted sections
        """
        if not DOCX_AVAILABLE:
            raise Exception("python-docx library is not installed. Please install it with: pip install python-docx")
        
        try:
            # Type ignore to satisfy linter since we've already checked availability
            doc = Document(file_path)
            text = ""
            # Type ignore for the same reason
            for paragraph in doc.paragraphs:  # type: ignore
                text += paragraph.text + "\n"
            
            return self._extract_sections(text)
        except Exception as e:
            raise Exception(f"Error parsing DOCX file: {str(e)}")
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract key sections from text using pattern matching
        
        Args:
            text (str): Full text of the document
            
        Returns:
            Dict[str, str]: Dictionary with extracted sections
        """
        # Convert to lowercase for matching
        lower_text = text.lower()
        
        # Initialize result dictionary
        sections = {
            'abstract': '',
            'objectives': '',
            'methodology': '',
            'budget': '',
            'outcomes': '',
            'full_text': text
        }
        
        # Find section positions
        section_positions = {}
        for section, patterns in self.section_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, lower_text, re.IGNORECASE))
                if matches:
                    # Use the first match as the section start
                    section_positions[section] = matches[0].start()
                    break
        
        # Sort sections by their position in the document
        sorted_sections = sorted(section_positions.items(), key=lambda x: x[1])
        
        # Extract content between sections
        for i, (section, position) in enumerate(sorted_sections):
            # Determine end position (next section or end of document)
            if i < len(sorted_sections) - 1:
                end_position = sorted_sections[i + 1][1]
            else:
                end_position = len(text)
            
            # Extract content
            content = text[position:end_position].strip()
            
            # Remove the section header
            lines = content.split('\n')
            if len(lines) > 1:
                # Try to remove the header line
                content = '\n'.join(lines[1:]).strip()
            
            sections[section] = content
        
        # If we couldn't find specific sections, use the full text as abstract
        if not any(sections[section] for section in ['abstract', 'objectives', 'methodology', 'budget', 'outcomes']):
            sections['abstract'] = text[:min(1000, len(text))]  # First 1000 characters as abstract
        
        return sections
    
    def convert_to_csv_format(self, sections: Dict[str, str], proposal_id: str, title: str = "") -> Dict[str, Union[str, float]]:
        """
        Convert parsed sections to CSV format compatible with the evaluator
        
        Args:
            sections (Dict[str, str]): Parsed sections from document
            proposal_id (str): Unique identifier for the proposal
            title (str): Title of the proposal (if available)
            
        Returns:
            Dict[str, Union[str, float]]: CSV-compatible format
        """
        # Use the first sentence of abstract as title if not provided
        if not title and sections['abstract']:
            # Extract first sentence as title
            sentences = re.split(r'[.!?]+', sections['abstract'])
            title = sentences[0].strip() if sentences else "Untitled Proposal"
        
        # Combine relevant sections for the abstract field
        abstract_parts = []
        if sections['abstract']:
            abstract_parts.append(sections['abstract'])
        if sections['objectives']:
            abstract_parts.append(f"Objectives: {sections['objectives']}")
        if sections['methodology']:
            abstract_parts.append(f"Methodology: {sections['methodology']}")
        if sections['outcomes']:
            abstract_parts.append(f"Expected Outcomes: {sections['outcomes']}")
        
        abstract = " ".join(abstract_parts)
        
        # Extract funding information from budget section if available
        funding = self._extract_funding(sections['budget'])
        
        return {
            'Proposal_ID': proposal_id,
            'Title': title,
            'Abstract': abstract,
            'Funding_Requested': funding
        }
    
    def _extract_funding(self, budget_text: str) -> float:
        """
        Extract funding amount from budget text
        
        Args:
            budget_text (str): Text containing budget information
            
        Returns:
            float: Extracted funding amount (default 0 if not found)
        """
        if not budget_text:
            return 0.0
        
        # Look for currency patterns
        patterns = [
            r'\$([0-9,]+\.?[0-9]*)',  # Dollar amounts
            r'([0-9,]+\.?[0-9]*)\s*dollars?',  # Dollar amounts spelled out
            r'₹([0-9,]+\.?[0-9]*)',  # Rupee amounts
            r'([0-9,]+\.?[0-9]*)\s*rupees?',  # Rupee amounts spelled out
            r'([0-9,]+\.?[0-9]*)'  # Generic numbers
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, budget_text, re.IGNORECASE)
            if matches:
                # Try to convert the first match to a number
                try:
                    # Remove commas and convert to float
                    amount = float(matches[0].replace(',', ''))
                    return amount
                except ValueError:
                    continue
        
        return 0.0

def main():
    """
    Main function to demonstrate document parsing
    """
    parser = DocumentParser()
    
    # Example usage (you would replace these paths with actual file paths)
    # For PDF:
    # sections = parser.parse_pdf("path/to/proposal.pdf")
    
    # For DOCX:
    # sections = parser.parse_docx("path/to/proposal.docx")
    
    # Convert to CSV format
    # csv_data = parser.convert_to_csv_format(sections, "PROP001", "AI for Mine Safety")
    
    print("DocumentParser module is ready for use.")

if __name__ == "__main__":
    main()