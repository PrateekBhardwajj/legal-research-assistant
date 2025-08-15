"""
Document processor for legal documents.
Handles parsing and chunking of contracts, case law, and statutes.
"""

import os
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import hashlib

import PyPDF2
from docx import Document
import pdfplumber
from bs4 import BeautifulSoup
import nltk
from loguru import logger


@dataclass
class DocumentChunk:
    """Represents a chunk of a legal document."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    document_id: str
    document_type: str
    section: str = None
    page_number: int = None
    
    
class LegalDocumentProcessor:
    """Processes various legal document formats with domain-specific chunking."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunk_size = config.get('chunk_size', 1000)
        self.chunk_overlap = config.get('chunk_overlap', 200)
        self.supported_formats = config.get('supported_formats', ['pdf', 'docx', 'txt', 'html'])
        
        # Legal document patterns
        self.section_patterns = {
            'contract': [
                r'(?i)(?:SECTION|CLAUSE|ARTICLE|TERM)\s+\d+',
                r'(?i)(?:WHEREAS|NOW THEREFORE|IN WITNESS WHEREOF)',
                r'(?i)(?:DEFINITIONS|TERMINATION|CONFIDENTIALITY|LIABILITY)'
            ],
            'case_law': [
                r'(?i)(?:FACTS|HOLDING|REASONING|CONCLUSION)',
                r'(?i)(?:PROCEDURAL HISTORY|ANALYSIS)',
                r'(?:\d+\s+[A-Z][a-z]+\s+\d+)' # Citation patterns
            ],
            'statute': [
                r'(?i)(?:§|Sec\.|Section)\s*\d+',
                r'(?i)(?:Art\.|Article)\s*\d+',
                r'(?i)(?:Ch\.|Chapter)\s*\d+'
            ]
        }
        
        # Initialize NLTK if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def process_document(self, file_path: str, document_type: str = None) -> List[DocumentChunk]:
        """Process a legal document and return chunks."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Determine document type if not provided
        if not document_type:
            document_type = self._detect_document_type(file_path)
        
        # Extract text based on file format
        text_content = self._extract_text(file_path)
        
        # Generate document metadata
        metadata = self._generate_metadata(file_path, document_type)
        
        # Apply document-specific chunking strategy
        chunks = self._chunk_document(text_content, document_type, metadata)
        
        logger.info(f"Processed {file_path.name}: {len(chunks)} chunks created")
        return chunks
    
    def _extract_text(self, file_path: Path) -> str:
        """Extract text content from various file formats."""
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.pdf':
                return self._extract_pdf_text(file_path)
            elif file_extension == '.docx':
                return self._extract_docx_text(file_path)
            elif file_extension == '.txt':
                return self._extract_txt_text(file_path)
            elif file_extension == '.html':
                return self._extract_html_text(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF using pdfplumber for better formatting."""
        text_content = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(f"[Page {page_num}]\n{page_text}\n")
        
        return '\n'.join(text_content)
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        doc = Document(file_path)
        paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
        return '\n'.join(paragraphs)
    
    def _extract_txt_text(self, file_path: Path) -> str:
        """Extract text from TXT file."""
        # Try different encodings to handle various file formats
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, read as binary and decode with errors ignored
        with open(file_path, 'rb') as file:
            content = file.read()
            return content.decode('utf-8', errors='ignore')
    
    def _extract_html_text(self, file_path: Path) -> str:
        """Extract text from HTML file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            return soup.get_text(separator='\n', strip=True)
    
    def _detect_document_type(self, file_path: Path) -> str:
        """Detect document type based on file path and content patterns."""
        file_name = file_path.name.lower()
        parent_dir = file_path.parent.name.lower()
        
        # Check parent directory
        if 'contract' in parent_dir:
            return 'contract'
        elif 'case' in parent_dir or 'law' in parent_dir:
            return 'case_law'
        elif 'statute' in parent_dir:
            return 'statute'
        
        # Check filename patterns
        if any(word in file_name for word in ['contract', 'agreement', 'terms']):
            return 'contract'
        elif any(word in file_name for word in ['case', 'court', 'judgment']):
            return 'case_law'
        elif any(word in file_name for word in ['statute', 'code', 'regulation']):
            return 'statute'
        
        return 'general'  # Default type
    
    def _generate_metadata(self, file_path: Path, document_type: str) -> Dict[str, Any]:
        """Generate metadata for the document."""
        return {
            'filename': file_path.name,
            'file_path': str(file_path),
            'document_type': document_type,
            'file_size': file_path.stat().st_size,
            'created_date': file_path.stat().st_ctime,
            'document_id': self._generate_document_id(file_path)
        }
    
    def _generate_document_id(self, file_path: Path) -> str:
        """Generate a unique document ID."""
        content = f"{file_path.name}{file_path.stat().st_size}{file_path.stat().st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _chunk_document(self, text: str, document_type: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Apply document-specific chunking strategies."""
        chunking_config = self.config.get('chunking_strategies', {}).get(document_type, {})
        method = chunking_config.get('method', 'paragraph_based')
        
        if method == 'section_based':
            return self._section_based_chunking(text, document_type, metadata)
        elif method == 'paragraph_based':
            return self._paragraph_based_chunking(text, document_type, metadata)
        elif method == 'hierarchical':
            return self._hierarchical_chunking(text, document_type, metadata)
        else:
            return self._default_chunking(text, document_type, metadata)
    
    def _section_based_chunking(self, text: str, document_type: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk document based on legal sections."""
        chunks = []
        sections = self._identify_sections(text, document_type)
        
        for i, (section_title, section_content) in enumerate(sections):
            if len(section_content.strip()) < 50:  # Skip very short sections
                continue
                
            chunk_id = f"{metadata['document_id']}_section_{i}"
            chunk = DocumentChunk(
                content=section_content,
                metadata=metadata.copy(),
                chunk_id=chunk_id,
                document_id=metadata['document_id'],
                document_type=document_type,
                section=section_title
            )
            chunks.append(chunk)
        
        return chunks
    
    def _paragraph_based_chunking(self, text: str, document_type: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk document based on paragraphs with overlap."""
        paragraphs = self._split_into_paragraphs(text)
        chunks = []
        
        current_chunk = ""
        chunk_count = 0
        
        for para in paragraphs:
            if len(current_chunk + para) > self.chunk_size and current_chunk:
                # Create chunk
                chunk_id = f"{metadata['document_id']}_chunk_{chunk_count}"
                chunk = DocumentChunk(
                    content=current_chunk.strip(),
                    metadata=metadata.copy(),
                    chunk_id=chunk_id,
                    document_id=metadata['document_id'],
                    document_type=document_type
                )
                chunks.append(chunk)
                
                # Handle overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n" + para
                chunk_count += 1
            else:
                current_chunk += "\n" + para if current_chunk else para
        
        # Add final chunk
        if current_chunk.strip():
            chunk_id = f"{metadata['document_id']}_chunk_{chunk_count}"
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                metadata=metadata.copy(),
                chunk_id=chunk_id,
                document_id=metadata['document_id'],
                document_type=document_type
            )
            chunks.append(chunk)
        
        return chunks
    
    def _hierarchical_chunking(self, text: str, document_type: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk document based on hierarchical structure (for statutes)."""
        hierarchy_markers = self.config.get('chunking_strategies', {}).get(document_type, {}).get('hierarchy_markers', [])
        
        # Split by hierarchy markers
        sections = []
        current_section = ""
        current_marker = None
        
        for line in text.split('\n'):
            line_marker = None
            for marker in hierarchy_markers:
                if marker in line:
                    line_marker = marker
                    break
            
            if line_marker and current_section:
                sections.append((current_marker, current_section.strip()))
                current_section = line
                current_marker = line_marker
            else:
                current_section += "\n" + line
        
        # Add final section
        if current_section.strip():
            sections.append((current_marker, current_section.strip()))
        
        # Convert to chunks
        chunks = []
        for i, (marker, content) in enumerate(sections):
            if len(content) < 50:  # Skip short sections
                continue
                
            chunk_id = f"{metadata['document_id']}_hierarchy_{i}"
            chunk = DocumentChunk(
                content=content,
                metadata=metadata.copy(),
                chunk_id=chunk_id,
                document_id=metadata['document_id'],
                document_type=document_type,
                section=marker
            )
            chunks.append(chunk)
        
        return chunks
    
    def _default_chunking(self, text: str, document_type: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Default chunking strategy with fixed size and overlap."""
        chunks = []
        text_length = len(text)
        chunk_count = 0
        
        for start in range(0, text_length, self.chunk_size - self.chunk_overlap):
            end = min(start + self.chunk_size, text_length)
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunk_id = f"{metadata['document_id']}_default_{chunk_count}"
                chunk = DocumentChunk(
                    content=chunk_text.strip(),
                    metadata=metadata.copy(),
                    chunk_id=chunk_id,
                    document_id=metadata['document_id'],
                    document_type=document_type
                )
                chunks.append(chunk)
                chunk_count += 1
        
        return chunks
    
    def _identify_sections(self, text: str, document_type: str) -> List[Tuple[str, str]]:
        """Identify sections in legal documents."""
        patterns = self.section_patterns.get(document_type, [])
        sections = []
        
        current_section_title = "Introduction"
        current_section_content = ""
        
        for line in text.split('\n'):
            # Check if line matches any section pattern
            is_section_header = False
            for pattern in patterns:
                if re.search(pattern, line):
                    # Save current section
                    if current_section_content.strip():
                        sections.append((current_section_title, current_section_content.strip()))
                    
                    # Start new section
                    current_section_title = line.strip()
                    current_section_content = ""
                    is_section_header = True
                    break
            
            if not is_section_header:
                current_section_content += line + "\n"
        
        # Add final section
        if current_section_content.strip():
            sections.append((current_section_title, current_section_content.strip()))
        
        return sections
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into meaningful paragraphs."""
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean and filter paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if len(para) > 20:  # Filter very short paragraphs
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def batch_process_documents(self, directory_path: str) -> List[DocumentChunk]:
        """Process all documents in a directory."""
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")
        
        all_chunks = []
        supported_extensions = [f".{fmt}" for fmt in self.supported_formats]
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    chunks = self.process_document(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue
        
        logger.info(f"Batch processed {len(all_chunks)} chunks from {directory_path}")
        return all_chunks
