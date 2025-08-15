# Multi-Document Legal Research Assistant

A Retrieval-Augmented Generation (RAG) system designed to analyze multiple legal documents and provide contextual answers to legal queries with proper citations.

## 🎯 Features

- **Multi-Format Document Processing**: Handles contracts, case law, statutes, and other legal documents
- **Contextual Legal Analysis**: Provides accurate answers with proper legal citations  
- **Conflict Resolution**: Identifies and handles conflicting information across documents
- **Section-Specific Referencing**: Precise citation to specific document sections
- **Legal Terminology Processing**: Understands domain-specific legal language
- **Interactive Web Interface**: User-friendly Streamlit application

## 🏗️ Architecture

```
├── app.py                 # Streamlit application
├── src/
│   ├── document_processor.py    # Document parsing and chunking
│   ├── embedding_manager.py     # Embedding generation and management
│   ├── vector_store.py          # Vector database operations
│   ├── retrieval_system.py      # Document retrieval logic
│   ├── legal_analyzer.py        # Legal analysis and conflict resolution
│   └── citation_manager.py      # Citation formatting and verification
├── data/
│   ├── contracts/              # Sample contracts
│   ├── case_law/              # Legal cases
│   └── statutes/              # Legal statutes
├── requirements.txt
├── config.yaml
└── tests/
```

## 🚀 Quick Start

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd legal-research-assistant
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Add your API keys (OpenAI, etc.)
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## 📋 Requirements

- Python 3.8+
- OpenAI API key (for embeddings and LLM)
- Sufficient disk space for document storage and vector database

## 🔧 Technical Implementation

### Embedding Strategy
- Uses OpenAI's `text-embedding-ada-002` for high-quality legal document embeddings
- Implements domain-specific embedding fine-tuning for legal terminology

### Chunking Strategy
- **Contract Documents**: Section-based chunking (clauses, terms, conditions)
- **Case Law**: Paragraph-based with legal reasoning preservation
- **Statutes**: Article and subsection-based chunking
- Maintains document hierarchy and cross-references

### Vector Database
- **Primary**: ChromaDB for local development
- **Alternative**: Pinecone for production deployment
- Implements hybrid search (semantic + keyword)

### Conflict Resolution
- Identifies contradictory information across documents
- Provides confidence scores for each source
- Highlights areas requiring legal professional review

## 📊 Evaluation Metrics

- **Retrieval Accuracy**: Precision@K, Recall@K
- **Response Quality**: RAGAS evaluation framework
- **Citation Accuracy**: Verification of source attributions
- **Latency**: Response time optimization
- **Legal Relevance**: Domain expert evaluation

## 🎯 Use Cases

1. **Contract Analysis**: "What are the termination clauses in my employment contracts?"
2. **Legal Research**: "Find precedents for intellectual property disputes"
3. **Compliance Review**: "What statutes apply to data privacy in healthcare?"
4. **Comparative Analysis**: "Compare liability terms across vendor agreements"

## 🔒 Important Disclaimers

⚠️ **This system is for research and educational purposes only. It does not constitute legal advice. Always consult with qualified legal professionals for legal matters.**

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for educational purposes as part of RAG system development
- Uses state-of-the-art NLP models for legal document processing
- Inspired by modern legal tech solutions
