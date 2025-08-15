"""
Simple test script to verify the legal research assistant system.
"""

import yaml
import os
from pathlib import Path

# Import our modules
from src.document_processor import LegalDocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStoreManager

def load_config():
    """Load configuration from YAML file."""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_document_processing():
    """Test document processing functionality."""
    print("🔍 Testing document processing...")
    
    config = load_config()
    doc_config = config.get('document_processing', {})
    
    processor = LegalDocumentProcessor(doc_config)
    
    # Test processing sample documents
    sample_files = [
        'data/contracts/employment_agreement.txt',
        'data/case_law/smith_v_jones_2023.txt'
    ]
    
    all_chunks = []
    for file_path in sample_files:
        if Path(file_path).exists():
            print(f"  📄 Processing {file_path}...")
            chunks = processor.process_document(file_path)
            print(f"    ✅ Created {len(chunks)} chunks")
            all_chunks.extend(chunks)
        else:
            print(f"  ❌ File not found: {file_path}")
    
    print(f"  📊 Total chunks processed: {len(all_chunks)}")
    return all_chunks

def test_embedding_generation(chunks):
    """Test embedding generation."""
    print("\n🔍 Testing embedding generation...")
    
    config = load_config()
    embed_config = config.get('openai', {})
    
    embedding_manager = EmbeddingManager(embed_config)
    
    # Test with first few chunks
    test_chunks = chunks[:2]  # Test with 2 chunks
    
    if test_chunks:
        print(f"  🧠 Generating embeddings for {len(test_chunks)} chunks...")
        embeddings = embedding_manager.generate_embeddings(test_chunks)
        print(f"    ✅ Generated {len(embeddings)} embeddings")
        
        if embeddings:
            print(f"    📏 Embedding dimension: {len(embeddings[0].embedding)}")
            print(f"    🏷️  Model used: {embeddings[0].model_name}")
        
        return embeddings
    else:
        print("  ❌ No chunks available for embedding test")
        return []

def test_vector_store(chunks, embeddings):
    """Test vector store functionality."""
    print("\n🔍 Testing vector store...")
    
    config = load_config()
    vector_config = config.get('vector_db', {})
    
    vector_store = VectorStoreManager(vector_config)
    
    if chunks and embeddings:
        print(f"  💾 Adding {len(chunks)} documents to vector store...")
        success = vector_store.add_documents(chunks, embeddings)
        
        if success:
            print("    ✅ Documents added successfully")
            
            # Test search
            print("  🔍 Testing search functionality...")
            query_embedding = embeddings[0].embedding  # Use first embedding as test query
            
            results = vector_store.search(query_embedding, top_k=2)
            print(f"    📊 Search returned {len(results)} results")
            
            if results:
                print(f"    🎯 Top result similarity: {results[0].get('similarity_score', 0):.3f}")
            
            return True
        else:
            print("    ❌ Failed to add documents")
            return False
    else:
        print("  ⚠️ No documents or embeddings available for testing")
        return False

def main():
    """Run system tests."""
    print("🚀 Legal Research Assistant - System Test")
    print("=" * 50)
    
    try:
        # Test 1: Document Processing
        chunks = test_document_processing()
        
        if not chunks:
            print("\n❌ No documents processed. Stopping tests.")
            return
        
        # Test 2: Embedding Generation
        embeddings = test_embedding_generation(chunks[:2])  # Test with 2 chunks
        
        if not embeddings:
            print("\n⚠️ No embeddings generated. Skipping vector store test.")
            print("💡 Note: This is normal if you haven't set an OpenAI API key.")
            print("   The system will use local embeddings instead.")
            return
        
        # Test 3: Vector Store
        vector_success = test_vector_store(chunks[:2], embeddings)
        
        if vector_success:
            print("\n🎉 All tests passed! System is ready to use.")
        else:
            print("\n⚠️ Some tests had issues, but core functionality works.")
        
        print("\n📋 Next steps:")
        print("1. Set up your OpenAI API key in .env file (for full functionality)")
        print("2. Run: streamlit run app.py")
        print("3. Upload legal documents and start searching!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("Check the logs above for more details.")

if __name__ == "__main__":
    main()
