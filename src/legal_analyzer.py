"""
Legal analyzer for AI-powered legal analysis and conflict detection.
"""

import openai
from typing import List, Dict, Any, Optional
import re
from loguru import logger


class LegalAnalyzer:
    """AI-powered legal document analysis."""
    
    def __init__(self, openai_config: Dict[str, Any], config: Dict[str, Any]):
        self.openai_config = openai_config
        self.config = config
        
        # Set up OpenAI
        self.api_key = openai_config.get('api_key')
        if self.api_key:
            openai.api_key = self.api_key
        
        self.model = openai_config.get('model', 'gpt-3.5-turbo')
        self.temperature = openai_config.get('temperature', 0.1)
        self.max_tokens = openai_config.get('max_tokens', 2000)
        
        # Analysis settings
        self.conflict_detection = config.get('conflict_detection', True)
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        
        logger.info("Legal analyzer initialized")
    
    def analyze_query_results(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze search results and provide legal insights."""
        
        if not self.api_key:
            return {
                "summary": "OpenAI API key required for legal analysis",
                "conflicts": [],
                "confidence": 0.0
            }
        
        if not results:
            return {
                "summary": "No documents found to analyze",
                "conflicts": [],
                "confidence": 0.0
            }
        
        try:
            # Prepare context from search results
            context = self._prepare_context(results)
            
            # Generate analysis
            analysis = self._generate_analysis(query, context)
            
            # Detect conflicts if enabled
            conflicts = []
            if self.conflict_detection and len(results) > 1:
                conflicts = self._detect_conflicts(results)
            
            return {
                "summary": analysis,
                "conflicts": conflicts,
                "confidence": self._calculate_confidence(results),
                "num_sources": len(results)
            }
            
        except Exception as e:
            logger.error(f"Legal analysis failed: {e}")
            return {
                "summary": f"Analysis failed: {e}",
                "conflicts": [],
                "confidence": 0.0
            }
    
    def _prepare_context(self, results: List[Dict[str, Any]], max_length: int = 3000) -> str:
        """Prepare context from search results for analysis."""
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results):
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            
            # Format source information
            source_info = f"[Source {i+1}: {metadata.get('filename', 'Unknown')} - {metadata.get('document_type', 'Unknown')}]"
            
            # Truncate content if needed
            available_length = max_length - current_length - len(source_info) - 100  # Buffer
            if available_length <= 0:
                break
            
            if len(content) > available_length:
                content = content[:available_length] + "..."
            
            context_part = f"{source_info}\n{content}\n"
            context_parts.append(context_part)
            current_length += len(context_part)
            
            if current_length >= max_length:
                break
        
        return "\n".join(context_parts)
    
    def _generate_analysis(self, query: str, context: str) -> str:
        """Generate AI-powered legal analysis."""
        
        system_prompt = """You are an expert legal research assistant. Provide concise, accurate analysis of legal documents based on the user's query. 

Guidelines:
- Provide clear, structured answers
- Cite specific sources when making statements
- Highlight key legal principles and precedents
- Be objective and avoid giving legal advice
- Use appropriate legal terminology
- Structure your response with clear headings when appropriate

Remember: This is for research purposes only and does not constitute legal advice."""

        user_prompt = f"""Query: {query}

Based on the following legal documents, provide a comprehensive analysis:

{context}

Please analyze the above documents in relation to the query and provide:
1. Summary of relevant findings
2. Key legal principles involved
3. How the sources relate to the query
4. Any important distinctions or nuances

Analysis:"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return f"Failed to generate analysis: {e}"
    
    def _detect_conflicts(self, results: List[Dict[str, Any]]) -> List[str]:
        """Detect potential conflicts between different sources."""
        
        if not self.api_key or len(results) < 2:
            return []
        
        try:
            # Prepare content for conflict detection
            sources = []
            for i, result in enumerate(results):
                content = result.get('content', '')
                metadata = result.get('metadata', {})
                
                # Truncate for conflict analysis
                if len(content) > 800:
                    content = content[:800] + "..."
                
                sources.append({
                    'id': i + 1,
                    'filename': metadata.get('filename', f'Source {i+1}'),
                    'content': content
                })
            
            conflicts = []
            
            # Compare each pair of sources
            for i in range(len(sources)):
                for j in range(i + 1, len(sources)):
                    conflict = self._analyze_conflict_pair(sources[i], sources[j])
                    if conflict:
                        conflicts.append(conflict)
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Conflict detection failed: {e}")
            return []
    
    def _analyze_conflict_pair(self, source1: Dict[str, Any], source2: Dict[str, Any]) -> Optional[str]:
        """Analyze two sources for potential conflicts."""
        
        prompt = f"""Compare these two legal document excerpts and identify any conflicts, contradictions, or inconsistencies:

Source 1 ({source1['filename']}):
{source1['content']}

Source 2 ({source2['filename']}):
{source2['content']}

Are there any direct conflicts, contradictions, or inconsistencies between these sources? If yes, briefly describe the conflict. If no, respond with "No conflict detected."

Conflict analysis:"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a legal expert identifying conflicts between legal documents. Be precise and objective."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            analysis = response.choices[0].message.content.strip()
            
            # Check if a conflict was detected
            if "no conflict" not in analysis.lower():
                return f"Conflict between {source1['filename']} and {source2['filename']}: {analysis}"
            
            return None
            
        except Exception as e:
            logger.error(f"Conflict pair analysis failed: {e}")
            return None
    
    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the analysis."""
        
        if not results:
            return 0.0
        
        # Base confidence on average similarity scores and number of sources
        avg_similarity = sum(r.get('similarity_score', 0) for r in results) / len(results)
        
        # Adjust for number of sources (more sources = higher confidence up to a point)
        source_factor = min(len(results) / 5.0, 1.0)  # Max benefit at 5 sources
        
        # Combine factors
        confidence = (avg_similarity * 0.7) + (source_factor * 0.3)
        
        return min(confidence, 1.0)
    
    def analyze_document_type(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a specific document type and extract key information."""
        
        document_type = metadata.get('document_type', 'general')
        
        if document_type == 'contract':
            return self._analyze_contract(content, metadata)
        elif document_type == 'case_law':
            return self._analyze_case_law(content, metadata)
        elif document_type == 'statute':
            return self._analyze_statute(content, metadata)
        else:
            return self._analyze_general_document(content, metadata)
    
    def _analyze_contract(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contract-specific elements."""
        
        if not self.api_key:
            return {"analysis": "OpenAI API key required"}
        
        prompt = f"""Analyze this contract excerpt and identify key elements:

{content[:1500]}...

Please identify and summarize:
1. Parties involved
2. Key terms and conditions
3. Important clauses (termination, liability, etc.)
4. Notable provisions or restrictions
5. Potential areas of concern

Contract Analysis:"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a contract analysis expert. Provide structured analysis of contract terms."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            return {
                "type": "contract",
                "analysis": response.choices[0].message.content.strip(),
                "confidence": 0.8
            }
            
        except Exception as e:
            return {"analysis": f"Contract analysis failed: {e}"}
    
    def _analyze_case_law(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze case law elements."""
        
        if not self.api_key:
            return {"analysis": "OpenAI API key required"}
        
        prompt = f"""Analyze this legal case excerpt:

{content[:1500]}...

Please identify and summarize:
1. Case name and court
2. Key facts
3. Legal issues presented
4. Holding/decision
5. Important legal principles or precedents
6. Significance of the ruling

Case Analysis:"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a legal case analysis expert. Provide structured analysis of court decisions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            return {
                "type": "case_law",
                "analysis": response.choices[0].message.content.strip(),
                "confidence": 0.8
            }
            
        except Exception as e:
            return {"analysis": f"Case law analysis failed: {e}"}
    
    def _analyze_statute(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze statute elements."""
        
        if not self.api_key:
            return {"analysis": "OpenAI API key required"}
        
        prompt = f"""Analyze this statute excerpt:

{content[:1500]}...

Please identify and summarize:
1. Statute title and section
2. Scope and applicability
3. Key requirements or prohibitions
4. Penalties or enforcement mechanisms
5. Important definitions
6. Related provisions or cross-references

Statute Analysis:"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a statutory analysis expert. Provide structured analysis of legal statutes."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            return {
                "type": "statute",
                "analysis": response.choices[0].message.content.strip(),
                "confidence": 0.8
            }
            
        except Exception as e:
            return {"analysis": f"Statute analysis failed: {e}"}
    
    def _analyze_general_document(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze general legal document."""
        
        return {
            "type": "general",
            "analysis": "General legal document analysis not yet implemented",
            "confidence": 0.5
        }
