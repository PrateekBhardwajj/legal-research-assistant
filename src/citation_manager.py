"""
Citation manager for formatting legal citations in various styles.
"""

from typing import Dict, Any, Optional
import re
from loguru import logger


class CitationManager:
    """Manages legal citation formatting in various styles."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.citation_formats = config.get('citation_formats', ['bluebook', 'alr', 'westlaw'])
        self.default_format = 'bluebook'
        
        logger.info("Citation manager initialized")
    
    def format_citation(self, metadata: Dict[str, Any], format_type: str = None) -> str:
        """Format a citation based on document metadata."""
        
        format_type = format_type or self.default_format
        
        if format_type == 'bluebook':
            return self._format_bluebook(metadata)
        elif format_type == 'alr':
            return self._format_alr(metadata)
        elif format_type == 'westlaw':
            return self._format_westlaw(metadata)
        else:
            return self._format_basic(metadata)
    
    def _format_bluebook(self, metadata: Dict[str, Any]) -> str:
        """Format citation in Bluebook style."""
        
        document_type = metadata.get('document_type', 'general')
        filename = metadata.get('filename', 'Unknown Document')
        section = metadata.get('section', '')
        page_number = metadata.get('page_number', '')
        
        if document_type == 'case_law':
            return self._format_bluebook_case(metadata)
        elif document_type == 'statute':
            return self._format_bluebook_statute(metadata)
        elif document_type == 'contract':
            return self._format_bluebook_contract(metadata)
        else:
            # Generic Bluebook format
            citation = filename
            if section:
                citation += f", {section}"
            if page_number:
                citation += f", at {page_number}"
            return citation
    
    def _format_bluebook_case(self, metadata: Dict[str, Any]) -> str:
        """Format case law citation in Bluebook style."""
        
        filename = metadata.get('filename', 'Unknown Case')
        section = metadata.get('section', '')
        
        # Try to extract case information from filename or content
        case_name = self._extract_case_name(filename)
        court_info = self._extract_court_info(metadata)
        year = self._extract_year(metadata)
        
        # Basic Bluebook case citation format: Case Name, Citation (Court Year)
        citation = case_name
        
        if court_info:
            citation += f", {court_info}"
        
        if year:
            citation += f" ({year})"
        
        if section:
            citation += f", {section}"
        
        return citation
    
    def _format_bluebook_statute(self, metadata: Dict[str, Any]) -> str:
        """Format statute citation in Bluebook style."""
        
        filename = metadata.get('filename', 'Unknown Statute')
        section = metadata.get('section', '')
        
        # Extract statute information
        statute_name = self._extract_statute_name(filename)
        jurisdiction = self._extract_jurisdiction(metadata)
        year = self._extract_year(metadata)
        
        # Basic Bluebook statute citation format: Title § Section (Jurisdiction Year)
        citation = statute_name
        
        if section:
            section_num = self._extract_section_number(section)
            if section_num:
                citation += f" § {section_num}"
        
        if jurisdiction and year:
            citation += f" ({jurisdiction} {year})"
        
        return citation
    
    def _format_bluebook_contract(self, metadata: Dict[str, Any]) -> str:
        """Format contract citation in Bluebook style."""
        
        filename = metadata.get('filename', 'Unknown Contract')
        section = metadata.get('section', '')
        date = self._extract_date(metadata)
        
        citation = filename
        
        if date:
            citation += f" ({date})"
        
        if section:
            citation += f", {section}"
        
        return citation
    
    def _format_alr(self, metadata: Dict[str, Any]) -> str:
        """Format citation in ALR (American Law Reports) style."""
        
        filename = metadata.get('filename', 'Unknown Document')
        section = metadata.get('section', '')
        page_number = metadata.get('page_number', '')
        
        citation = filename
        
        if section:
            section_num = self._extract_section_number(section)
            citation += f" § {section_num or 'N/A'}"
        
        if page_number:
            citation += f", at {page_number}"
        
        return citation
    
    def _format_westlaw(self, metadata: Dict[str, Any]) -> str:
        """Format citation in Westlaw style."""
        
        filename = metadata.get('filename', 'Unknown Document')
        section = metadata.get('section', '')
        document_type = metadata.get('document_type', 'general')
        
        citation = filename
        
        if document_type == 'case_law':
            # Add WL identifier if available (would be extracted from metadata)
            wl_identifier = self._extract_westlaw_identifier(metadata)
            if wl_identifier:
                citation += f", {wl_identifier}"
        
        if section:
            citation += f", {section}"
        
        return citation
    
    def _format_basic(self, metadata: Dict[str, Any]) -> str:
        """Format basic citation."""
        
        filename = metadata.get('filename', 'Unknown Document')
        section = metadata.get('section', '')
        page_number = metadata.get('page_number', '')
        
        citation = filename
        
        if section:
            citation += f" ({section})"
        
        if page_number:
            citation += f", p. {page_number}"
        
        return citation
    
    def _extract_case_name(self, filename: str) -> str:
        """Extract case name from filename."""
        
        # Remove file extension
        name = re.sub(r'\.[^.]+$', '', filename)
        
        # Look for case name patterns (Party v. Party)
        case_pattern = r'([A-Z][a-zA-Z\s&]+)\s+v\.?\s+([A-Z][a-zA-Z\s&]+)'
        match = re.search(case_pattern, name)
        
        if match:
            plaintiff = match.group(1).strip()
            defendant = match.group(2).strip()
            return f"{plaintiff} v. {defendant}"
        
        # If no clear pattern, clean up the filename
        name = re.sub(r'[_-]', ' ', name)
        name = re.sub(r'\s+', ' ', name)
        return name.title()
    
    def _extract_court_info(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract court information from metadata or content."""
        
        filename = metadata.get('filename', '')
        
        # Look for court indicators in filename
        court_patterns = {
            'supreme': 'U.S.',
            'district': 'D.',
            'circuit': 'Cir.',
            'court of appeals': 'App.',
            'federal': 'F.',
            'state': 'State'
        }
        
        filename_lower = filename.lower()
        for pattern, abbreviation in court_patterns.items():
            if pattern in filename_lower:
                return abbreviation
        
        return None
    
    def _extract_year(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract year from metadata."""
        
        filename = metadata.get('filename', '')
        
        # Look for 4-digit years
        year_pattern = r'(19|20)\d{2}'
        match = re.search(year_pattern, filename)
        
        if match:
            return match.group(0)
        
        # Could also check file creation date as fallback
        created_date = metadata.get('created_date')
        if created_date:
            try:
                import datetime
                if isinstance(created_date, (int, float)):
                    dt = datetime.datetime.fromtimestamp(created_date)
                    return str(dt.year)
            except:
                pass
        
        return None
    
    def _extract_statute_name(self, filename: str) -> str:
        """Extract statute name from filename."""
        
        # Remove file extension
        name = re.sub(r'\.[^.]+$', '', filename)
        
        # Clean up common patterns
        name = re.sub(r'[_-]', ' ', name)
        name = re.sub(r'\s+', ' ', name)
        
        # Look for common statute patterns
        if 'usc' in name.lower():
            return "U.S.C."
        elif 'code' in name.lower():
            return name.title()
        elif 'statute' in name.lower():
            return name.title()
        
        return name.title()
    
    def _extract_jurisdiction(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract jurisdiction from metadata."""
        
        filename = metadata.get('filename', '').lower()
        
        # Common jurisdiction patterns
        jurisdictions = {
            'federal': 'Fed.',
            'california': 'Cal.',
            'new york': 'N.Y.',
            'texas': 'Tex.',
            'florida': 'Fla.',
            'illinois': 'Ill.',
            'delaware': 'Del.'
        }
        
        for jurisdiction, abbrev in jurisdictions.items():
            if jurisdiction in filename:
                return abbrev
        
        return None
    
    def _extract_section_number(self, section: str) -> Optional[str]:
        """Extract section number from section text."""
        
        if not section:
            return None
        
        # Look for section numbers
        patterns = [
            r'(?:§|Section|Sec\.)\s*(\d+(?:\.\d+)*)',
            r'(?:Article|Art\.)\s*(\d+)',
            r'(?:Chapter|Ch\.)\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, section, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Look for standalone numbers
        number_match = re.search(r'\b(\d+(?:\.\d+)*)\b', section)
        if number_match:
            return number_match.group(1)
        
        return None
    
    def _extract_date(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract date from metadata."""
        
        # This would be more sophisticated in a real implementation
        # For now, try to extract year
        return self._extract_year(metadata)
    
    def _extract_westlaw_identifier(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract Westlaw identifier if available."""
        
        # In a real implementation, this would look for WL identifiers
        # in the metadata or document content
        filename = metadata.get('filename', '')
        
        wl_pattern = r'(\d{4}\s+WL\s+\d+)'
        match = re.search(wl_pattern, filename)
        
        if match:
            return match.group(1)
        
        return None
    
    def batch_format_citations(self, results: list, format_type: str = None) -> list:
        """Format citations for a batch of search results."""
        
        formatted_results = []
        
        for result in results:
            metadata = result.get('metadata', {})
            citation = self.format_citation(metadata, format_type)
            
            # Add citation to result
            result_with_citation = result.copy()
            result_with_citation['citation'] = citation
            formatted_results.append(result_with_citation)
        
        return formatted_results
    
    def validate_citation(self, citation: str, format_type: str) -> Dict[str, Any]:
        """Validate a citation format."""
        
        validation_result = {
            'valid': True,
            'format': format_type,
            'issues': []
        }
        
        if not citation or citation.strip() == '':
            validation_result['valid'] = False
            validation_result['issues'].append('Citation is empty')
            return validation_result
        
        # Basic validation based on format type
        if format_type == 'bluebook':
            validation_result = self._validate_bluebook_citation(citation)
        elif format_type == 'alr':
            validation_result = self._validate_alr_citation(citation)
        elif format_type == 'westlaw':
            validation_result = self._validate_westlaw_citation(citation)
        
        return validation_result
    
    def _validate_bluebook_citation(self, citation: str) -> Dict[str, Any]:
        """Validate Bluebook citation format."""
        
        return {
            'valid': True,
            'format': 'bluebook',
            'issues': []  # Basic validation - could be enhanced
        }
    
    def _validate_alr_citation(self, citation: str) -> Dict[str, Any]:
        """Validate ALR citation format."""
        
        return {
            'valid': True,
            'format': 'alr',
            'issues': []
        }
    
    def _validate_westlaw_citation(self, citation: str) -> Dict[str, Any]:
        """Validate Westlaw citation format."""
        
        return {
            'valid': True,
            'format': 'westlaw',
            'issues': []
        }
