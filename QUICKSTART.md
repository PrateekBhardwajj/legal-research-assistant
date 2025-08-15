# Quick Start Guide - Legal Research Assistant

## 🚀 Getting Started in 5 Minutes

### Step 1: Setup Environment

```bash
# Run the setup script
python setup.py
```

This will:
- Install all Python dependencies
- Create necessary directories
- Create a template `.env` file
- Download required NLTK data

### Step 2: Configure API Keys

Edit the `.env` file with your API keys:

```bash
# API Keys (Required)
OPENAI_API_KEY=sk-your-actual-openai-key-here
PINECONE_API_KEY=your-pinecone-key-here  # Optional, uses ChromaDB by default
```

**Getting API Keys:**
- **OpenAI**: Sign up at [openai.com](https://openai.com) → Account → API Keys
- **Pinecone**: Sign up at [pinecone.io](https://pinecone.io) → API Keys (optional)

### Step 3: Launch the Application

```bash
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`

### Step 4: Upload Documents

1. Go to the **"📄 Document Management"** tab
2. Click **"Choose legal documents"**
3. Upload sample documents (PDF, DOCX, TXT):
   - We've included sample documents in `data/contracts/` and `data/case_law/`
4. Click **"📤 Process Documents"**

### Step 5: Start Searching

1. Go to the **"🔍 Search"** tab
2. Enter a legal query, for example:
   - "What are the termination clauses in employment contracts?"
   - "How are liquidated damages enforced?"
   - "What constitutes a material breach of contract?"
3. Click **"🔍 Search"**

## 📋 Sample Queries to Try

Once you've uploaded the sample documents, try these queries:

### Contract Analysis
- "What are the non-compete restrictions?"
- "How much notice is required for termination?"
- "What are the confidentiality obligations?"

### Case Law Research
- "What is the test for material breach of contract?"
- "When are liquidated damages enforceable?"
- "How does 'time is of the essence' affect contract performance?"

## 🔧 Configuration Options

### Vector Database Choice
- **ChromaDB** (default): Local, no setup required
- **Pinecone**: Cloud-based, requires API key and setup

Edit `config.yaml` to switch:
```yaml
vector_db:
  provider: "chromadb"  # or "pinecone"
```

### Model Configuration
Edit `config.yaml` to customize:
```yaml
openai:
  model: "gpt-3.5-turbo"  # or "gpt-4"
  embedding_model: "text-embedding-ada-002"
  temperature: 0.1
```

## 🎯 Features Walkthrough

### 1. Search Interface
- **Hybrid Search**: Combines semantic and keyword matching
- **Filters**: Filter by document type (contract, case_law, statute)
- **Advanced Options**: Adjust similarity thresholds, enable conflict detection

### 2. Document Management
- **Multi-format Support**: PDF, DOCX, TXT
- **Automatic Processing**: Intelligent chunking based on document type
- **Progress Tracking**: Real-time upload and processing status

### 3. Legal Analysis
- **Contextual Summaries**: AI-powered analysis of search results
- **Conflict Detection**: Identifies contradictory information
- **Citation Formatting**: Multiple citation formats (Bluebook, ALR, Westlaw)

### 4. Analytics Dashboard
- **Document Statistics**: Overview of your legal document corpus
- **Search Performance**: Metrics and insights
- **Usage Tracking**: Query history and patterns

## ⚠️ Important Notes

### Legal Disclaimer
This system is for **research and educational purposes only**. It does not constitute legal advice. Always consult qualified legal professionals for legal matters.

### API Usage
- OpenAI API usage will incur costs based on tokens processed
- Start with small document sets to understand costs
- Monitor usage in your OpenAI dashboard

### Performance Tips
- **Small Documents**: Start with 1-5 documents to test functionality
- **Chunking**: Larger documents are automatically split for better retrieval
- **Caching**: Search results are cached for faster subsequent queries

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**API Key Errors**
- Ensure API keys are properly set in `.env`
- Check API key validity in your provider dashboard
- Verify sufficient API credits

**ChromaDB Issues**
```bash
# Delete and recreate ChromaDB
rm -rf chromadb/
# Restart the application
```

**No Search Results**
- Check if documents were successfully processed
- Try broader search terms
- Lower the similarity threshold in advanced options

### Getting Help

1. **Check Logs**: Look for error messages in the Streamlit interface
2. **GitHub Issues**: Report bugs and request features
3. **Documentation**: Refer to the full README.md for detailed information

## 🎉 Success!

You now have a fully functional Legal Research Assistant! 

**Next Steps:**
- Upload your own legal documents
- Experiment with different search strategies
- Explore the analytics dashboard
- Customize settings for your use case

Happy researching! ⚖️
