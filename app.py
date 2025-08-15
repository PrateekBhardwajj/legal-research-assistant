"""
Main Streamlit application for the Legal Research Assistant.
"""

import streamlit as st
import os
import yaml
from pathlib import Path
from typing import List, Dict, Any
import time
import pandas as pd

from dotenv import load_dotenv
from loguru import logger

# Import our modules
from src.document_processor import LegalDocumentProcessor, DocumentChunk
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.retrieval_system import RetrievalSystem
from src.legal_analyzer import LegalAnalyzer
from src.citation_manager import CitationManager


class LegalResearchApp:
    """Main application class for the Legal Research Assistant."""
    
    def __init__(self):
        self.config = self.load_config()
        self.setup_logging()
        self.initialize_components()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            # Load environment variables
            load_dotenv()
            
            # Replace environment variables in config
            if 'openai' in config:
                config['openai']['api_key'] = os.getenv('OPENAI_API_KEY', config['openai'].get('api_key', ''))
            
            if 'vector_db' in config and 'pinecone' in config['vector_db']:
                config['vector_db']['pinecone']['api_key'] = os.getenv('PINECONE_API_KEY', 
                                                                      config['vector_db']['pinecone'].get('api_key', ''))
            
            return config
        except Exception as e:
            st.error(f"Failed to load configuration: {e}")
            return {}
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        logger.remove()  # Remove default handler
        logger.add(
            sink=lambda message: st.sidebar.write(f"🔧 {message}"),
            level=log_level,
            format="{time:HH:mm:ss} | {level} | {message}"
        )
    
    def initialize_components(self):
        """Initialize all system components."""
        try:
            # Initialize document processor
            doc_config = self.config.get('document_processing', {})
            self.document_processor = LegalDocumentProcessor(doc_config)
            
            # Initialize embedding manager
            embed_config = self.config.get('openai', {})
            self.embedding_manager = EmbeddingManager(embed_config)
            
            # Initialize vector store
            vector_config = self.config.get('vector_db', {})
            self.vector_store = VectorStoreManager(vector_config)
            
            # Initialize retrieval system
            retrieval_config = self.config.get('retrieval', {})
            self.retrieval_system = RetrievalSystem(
                embedding_manager=self.embedding_manager,
                vector_store=self.vector_store,
                config=retrieval_config
            )
            
            # Initialize legal analyzer
            analysis_config = self.config.get('legal_analysis', {})
            self.legal_analyzer = LegalAnalyzer(
                openai_config=self.config.get('openai', {}),
                config=analysis_config
            )
            
            # Initialize citation manager
            self.citation_manager = CitationManager(analysis_config)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            st.error(f"Failed to initialize components: {e}")
            st.stop()
    
    def run(self):
        """Run the Streamlit application."""
        # Page configuration
        st.set_page_config(
            page_title=self.config.get('streamlit', {}).get('page_title', 'Legal Research Assistant'),
            page_icon=self.config.get('streamlit', {}).get('page_icon', '⚖️'),
            layout=self.config.get('streamlit', {}).get('layout', 'wide')
        )
        
        # Main interface
        self.render_header()
        self.render_sidebar()
        
        # Main content area
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = 'search'
        
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📄 Document Management", "📊 Analytics", "⚙️ Settings"])
        
        with tab1:
            self.render_search_interface()
        
        with tab2:
            self.render_document_management()
        
        with tab3:
            self.render_analytics()
        
        with tab4:
            self.render_settings()
    
    def render_header(self):
        """Render the application header."""
        st.title("⚖️ Multi-Document Legal Research Assistant")
        st.markdown("""
        Advanced RAG system for legal document analysis with contextual answers, 
        proper citations, and conflict resolution capabilities.
        """)
        
        # Display system status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            doc_count = self.vector_store.get_document_count()
            st.metric("Documents", doc_count)
        
        with col2:
            st.metric("Vector Store", self.vector_store.provider.title())
        
        with col3:
            embedding_model = self.config.get('openai', {}).get('embedding_model', 'N/A')
            st.metric("Embedding Model", embedding_model.split('-')[-1].upper())
        
        with col4:
            llm_model = self.config.get('openai', {}).get('model', 'N/A')
            st.metric("LLM Model", llm_model.upper())
    
    def render_sidebar(self):
        """Render the sidebar with system information."""
        with st.sidebar:
            st.header("System Status")
            
            # Vector store info
            store_info = self.vector_store.get_store_info()
            st.json(store_info)
            
            st.header("Quick Actions")
            
            if st.button("🔄 Refresh System"):
                st.rerun()
            
            if st.button("🧹 Clear Cache"):
                st.cache_data.clear()
                st.success("Cache cleared!")
            
            # Recent activity (placeholder)
            st.header("Recent Activity")
            st.info("No recent activity")
    
    def render_search_interface(self):
        """Render the main search interface."""
        st.header("Legal Document Search & Analysis")
        
        # Search input
        query = st.text_area(
            "Enter your legal query:",
            placeholder="e.g., What are the termination clauses in employment contracts?",
            height=100
        )
        
        # Search options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_type = st.selectbox(
                "Search Type",
                ["Hybrid", "Semantic", "Keyword"],
                help="Hybrid combines semantic and keyword search"
            )
        
        with col2:
            document_filter = st.multiselect(
                "Document Types",
                ["contract", "case_law", "statute", "general"],
                default=[]
            )
        
        with col3:
            max_results = st.slider("Max Results", 1, 20, 5)
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.1)
                enable_conflict_detection = st.checkbox("Enable Conflict Detection", True)
            with col2:
                citation_format = st.selectbox("Citation Format", ["bluebook", "alr", "westlaw"])
                include_analysis = st.checkbox("Include Legal Analysis", True)
        
        # Search button
        if st.button("🔍 Search", type="primary") and query:
            self.perform_search(
                query, search_type, document_filter, max_results,
                similarity_threshold, enable_conflict_detection,
                citation_format, include_analysis
            )
    
    def perform_search(self, query: str, search_type: str, document_filter: List[str],
                      max_results: int, similarity_threshold: float,
                      enable_conflict_detection: bool, citation_format: str,
                      include_analysis: bool):
        """Perform the search and display results."""
        
        with st.spinner("Searching legal documents..."):
            try:
                # Prepare filters
                filters = {}
                if document_filter:
                    filters['document_type'] = document_filter
                
                # Perform search based on type
                if search_type == "Hybrid":
                    results = self.retrieval_system.hybrid_search(
                        query=query,
                        top_k=max_results,
                        filters=filters,
                        similarity_threshold=similarity_threshold
                    )
                elif search_type == "Semantic":
                    results = self.retrieval_system.semantic_search(
                        query=query,
                        top_k=max_results,
                        filters=filters,
                        similarity_threshold=similarity_threshold
                    )
                else:  # Keyword
                    results = self.retrieval_system.keyword_search(
                        query=query,
                        top_k=max_results,
                        filters=filters
                    )
                
                if not results:
                    st.warning("No relevant documents found. Try adjusting your query or search parameters.")
                    return
                
                # Display results
                st.success(f"Found {len(results)} relevant documents")
                
                # Legal analysis if requested
                if include_analysis:
                    with st.expander("📋 Legal Analysis Summary", expanded=True):
                        analysis = self.legal_analyzer.analyze_query_results(query, results)
                        st.markdown(analysis.get('summary', 'No analysis available'))
                        
                        if enable_conflict_detection and analysis.get('conflicts'):
                            st.warning("⚠️ Potential conflicts detected between sources")
                            for conflict in analysis['conflicts']:
                                st.markdown(f"- {conflict}")
                
                # Display individual results
                for i, result in enumerate(results, 1):
                    with st.expander(f"📄 Result {i} - {result['metadata'].get('filename', 'Unknown')} (Score: {result.get('similarity_score', 0):.3f})"):
                        
                        # Document metadata
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Document Type:** {result['metadata'].get('document_type', 'N/A')}")
                            st.markdown(f"**Section:** {result['metadata'].get('section', 'N/A')}")
                        with col2:
                            st.markdown(f"**File:** {result['metadata'].get('filename', 'N/A')}")
                            st.markdown(f"**Page:** {result['metadata'].get('page_number', 'N/A')}")
                        
                        # Content
                        st.markdown("**Content:**")
                        st.markdown(result['content'])
                        
                        # Citation
                        if citation_format:
                            citation = self.citation_manager.format_citation(
                                result['metadata'], citation_format
                            )
                            st.markdown(f"**Citation:** {citation}")
            
            except Exception as e:
                st.error(f"Search failed: {e}")
                logger.error(f"Search error: {e}")
    
    def render_document_management(self):
        """Render document management interface."""
        st.header("Document Management")
        
        # File upload
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose legal documents",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        if uploaded_files:
            if st.button("📤 Process Documents"):
                self.process_uploaded_documents(uploaded_files)
        
        # Existing documents
        st.subheader("Existing Documents")
        
        # Display document statistics
        store_info = self.vector_store.get_store_info()
        if store_info.get('document_count', 0) > 0:
            st.info(f"Currently managing {store_info['document_count']} document chunks")
            
            # Option to delete all documents
            if st.button("🗑️ Clear All Documents", type="secondary"):
                if st.confirm("Are you sure you want to delete all documents?"):
                    # This would need implementation in vector store
                    st.warning("Clear all functionality needs to be implemented")
        else:
            st.info("No documents currently loaded. Upload some documents to get started!")
    
    def process_uploaded_documents(self, uploaded_files):
        """Process uploaded documents."""
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            all_chunks = []
            
            # Process each file
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save temporary file
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # Process document
                    chunks = self.document_processor.process_document(temp_path)
                    all_chunks.extend(chunks)
                    
                    st.success(f"✅ Processed {uploaded_file.name}: {len(chunks)} chunks")
                    
                except Exception as e:
                    st.error(f"❌ Failed to process {uploaded_file.name}: {e}")
                
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            if all_chunks:
                status_text.text("Generating embeddings...")
                
                # Generate embeddings
                embeddings = self.embedding_manager.batch_generate_embeddings(all_chunks)
                
                status_text.text("Storing in vector database...")
                
                # Store in vector database
                success = self.vector_store.add_documents(all_chunks, embeddings)
                
                if success:
                    st.success(f"🎉 Successfully processed and stored {len(all_chunks)} document chunks!")
                else:
                    st.error("Failed to store documents in vector database")
            
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            
        except Exception as e:
            st.error(f"Document processing failed: {e}")
            logger.error(f"Document processing error: {e}")
    
    def render_analytics(self):
        """Render analytics and metrics."""
        st.header("System Analytics")
        
        # Vector store analytics
        store_info = self.vector_store.get_store_info()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Document Statistics")
            
            # Create sample data for demonstration
            doc_types = ['Contract', 'Case Law', 'Statute', 'General']
            doc_counts = [25, 15, 10, 5]  # Placeholder data
            
            df = pd.DataFrame({
                'Document Type': doc_types,
                'Count': doc_counts
            })
            
            st.bar_chart(df.set_index('Document Type'))
        
        with col2:
            st.subheader("Search Performance")
            
            # Placeholder metrics
            st.metric("Avg Response Time", "1.2s", "-0.3s")
            st.metric("Cache Hit Rate", "78%", "+5%")
            st.metric("Embedding Quality", "0.85", "+0.02")
        
        st.subheader("Recent Queries")
        st.info("Query history tracking would be implemented here")
    
    def render_settings(self):
        """Render settings interface."""
        st.header("System Settings")
        
        # Model settings
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_llm_model = st.selectbox(
                "LLM Model",
                ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                index=0
            )
            
            new_temperature = st.slider(
                "Temperature",
                0.0, 1.0, 0.1, 0.1
            )
        
        with col2:
            new_embedding_model = st.selectbox(
                "Embedding Model",
                ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
                index=0
            )
            
            new_max_tokens = st.slider(
                "Max Tokens",
                500, 4000, 2000, 100
            )
        
        # Search settings
        st.subheader("Search Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_similarity_threshold = st.slider(
                "Default Similarity Threshold",
                0.0, 1.0, 0.75, 0.05
            )
            
            new_max_results = st.slider(
                "Default Max Results",
                1, 50, 5
            )
        
        with col2:
            new_semantic_weight = st.slider(
                "Semantic Search Weight",
                0.0, 1.0, 0.7, 0.1
            )
            
            new_keyword_weight = st.slider(
                "Keyword Search Weight",
                0.0, 1.0, 0.3, 0.1
            )
        
        # API Keys section
        st.subheader("API Configuration")
        
        with st.expander("API Keys (Environment Variables)"):
            st.code("""
            OPENAI_API_KEY=your_openai_api_key_here
            PINECONE_API_KEY=your_pinecone_api_key_here
            """)
            
            st.info("API keys are loaded from environment variables for security.")
        
        # Save settings button
        if st.button("💾 Save Settings"):
            st.success("Settings saved! (Implementation needed)")


# Create placeholder modules if they don't exist
def create_placeholder_modules():
    """Create placeholder modules for missing components."""
    
    # Retrieval System
    retrieval_path = Path("src/retrieval_system.py")
    if not retrieval_path.exists():
        retrieval_code = '''
"""Placeholder retrieval system."""
from typing import List, Dict, Any

class RetrievalSystem:
    def __init__(self, embedding_manager, vector_store, config):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.config = config
    
    def hybrid_search(self, query: str, top_k: int = 5, filters: Dict = None, similarity_threshold: float = 0.75):
        query_embedding = self.embedding_manager.embed_query(query)
        return self.vector_store.hybrid_search(query_embedding, query, top_k)
    
    def semantic_search(self, query: str, top_k: int = 5, filters: Dict = None, similarity_threshold: float = 0.75):
        query_embedding = self.embedding_manager.embed_query(query)
        return self.vector_store.search(query_embedding, top_k, filters)
    
    def keyword_search(self, query: str, top_k: int = 5, filters: Dict = None):
        # Simplified keyword search
        query_embedding = self.embedding_manager.embed_query(query)
        return self.vector_store.search(query_embedding, top_k, filters)
'''
        with open(retrieval_path, 'w') as f:
            f.write(retrieval_code)
    
    # Legal Analyzer
    analyzer_path = Path("src/legal_analyzer.py")
    if not analyzer_path.exists():
        analyzer_code = '''
"""Placeholder legal analyzer."""
import openai
from typing import List, Dict, Any

class LegalAnalyzer:
    def __init__(self, openai_config, config):
        self.openai_config = openai_config
        self.config = config
        if openai_config.get('api_key'):
            openai.api_key = openai_config['api_key']
    
    def analyze_query_results(self, query: str, results: List[Dict]) -> Dict[str, Any]:
        if not self.openai_config.get('api_key'):
            return {"summary": "OpenAI API key required for legal analysis"}
        
        try:
            context = "\\n\\n".join([r['content'][:500] + "..." for r in results[:3]])
            
            response = openai.ChatCompletion.create(
                model=self.openai_config.get('model', 'gpt-3.5-turbo'),
                messages=[
                    {"role": "system", "content": "You are a legal research assistant. Provide concise analysis of legal documents."},
                    {"role": "user", "content": f"Query: {query}\\n\\nRelevant documents:\\n{context}\\n\\nProvide a brief analysis:"}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            return {
                "summary": response.choices[0].message.content,
                "conflicts": []
            }
        except Exception as e:
            return {"summary": f"Analysis failed: {e}", "conflicts": []}
'''
        with open(analyzer_path, 'w') as f:
            f.write(analyzer_code)
    
    # Citation Manager
    citation_path = Path("src/citation_manager.py")
    if not citation_path.exists():
        citation_code = '''
"""Placeholder citation manager."""
from typing import Dict, Any

class CitationManager:
    def __init__(self, config):
        self.config = config
    
    def format_citation(self, metadata: Dict[str, Any], format_type: str = "bluebook") -> str:
        filename = metadata.get('filename', 'Unknown Document')
        section = metadata.get('section', '')
        page = metadata.get('page_number', '')
        
        if format_type == "bluebook":
            citation = filename
            if section:
                citation += f", {section}"
            if page:
                citation += f", at {page}"
            return citation
        elif format_type == "alr":
            return f"{filename} § {section or 'N/A'}"
        elif format_type == "westlaw":
            return f"{filename}, {section or 'N/A'}"
        else:
            return f"{filename} ({section or 'N/A'})"
'''
        with open(citation_path, 'w') as f:
            f.write(citation_code)


def main():
    """Main entry point for the application."""
    
    # Create placeholder modules if they don't exist
    create_placeholder_modules()
    
    # Initialize and run the app
    app = LegalResearchApp()
    app.run()


if __name__ == "__main__":
    main()
