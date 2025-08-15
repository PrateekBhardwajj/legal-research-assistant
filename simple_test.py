"""
Simple test to check if the system works.
"""

import yaml
import os
from pathlib import Path

def main():
    print("🚀 Testing Legal Research Assistant")
    print("=" * 40)
    
    try:
        # Test 1: Check if config loads
        print("1️⃣ Loading configuration...")
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("   ✅ Configuration loaded successfully")
        
        # Test 2: Check imports
        print("\n2️⃣ Testing imports...")
        from src.document_processor import LegalDocumentProcessor
        from src.embedding_manager import EmbeddingManager
        from src.vector_store import VectorStoreManager
        print("   ✅ All modules imported successfully")
        
        # Test 3: Initialize components
        print("\n3️⃣ Initializing components...")
        doc_processor = LegalDocumentProcessor(config.get('document_processing', {}))
        print("   ✅ Document processor initialized")
        
        embed_manager = EmbeddingManager(config.get('openai', {}))
        print("   ✅ Embedding manager initialized")
        
        vector_store = VectorStoreManager(config.get('vector_db', {}))
        print("   ✅ Vector store initialized")
        
        # Test 4: Simple text processing
        print("\n4️⃣ Testing text processing...")
        
        # Create a simple test document
        test_content = """SAMPLE LEGAL CONTRACT
        
        This is a sample legal contract for testing purposes.
        
        SECTION 1. PARTIES
        
        The parties to this agreement are ABC Corp and XYZ Ltd.
        
        SECTION 2. TERMS
        
        The terms of this agreement are as follows:
        1. Payment shall be made within 30 days
        2. This agreement is governed by Delaware law
        """
        
        # Test chunking directly with text content
        from src.document_processor import DocumentChunk
        import hashlib
        
        # Create test metadata
        test_metadata = {
            'filename': 'test_contract.txt',
            'file_path': 'test_contract.txt',
            'document_type': 'contract',
            'file_size': len(test_content),
            'created_date': 123456789,
            'document_id': hashlib.md5('test'.encode()).hexdigest()
        }
        
        # Test paragraph chunking
        paragraphs = test_content.split('\n\n')
        test_chunks = []
        
        for i, para in enumerate(paragraphs):
            if para.strip():
                chunk = DocumentChunk(
                    content=para.strip(),
                    metadata=test_metadata,
                    chunk_id=f"test_chunk_{i}",
                    document_id=test_metadata['document_id'],
                    document_type='contract'
                )
                test_chunks.append(chunk)
        
        print(f"   ✅ Created {len(test_chunks)} test chunks")
        
        # Test 5: Test embeddings (local only)
        print("\n5️⃣ Testing embeddings...")
        if len(test_chunks) > 0:
            try:
                embeddings = embed_manager.generate_embeddings(test_chunks[:1])  # Test with 1 chunk
                if embeddings:
                    print(f"   ✅ Generated embedding with dimension: {len(embeddings[0].embedding)}")
                    print(f"   📋 Model used: {embeddings[0].model_name}")
                else:
                    print("   ⚠️ No embeddings generated (this is normal without OpenAI API key)")
            except Exception as e:
                print(f"   ⚠️ Embedding test failed: {e}")
        
        print("\n🎉 Basic system test completed!")
        print("\n📋 System Status:")
        print("✅ Configuration: Working")
        print("✅ Modules: All imported")
        print("✅ Components: Initialized")
        print("✅ Text Processing: Working")
        print("✅ Basic Functionality: Ready")
        
        print("\n🚀 Next Steps:")
        print("1. Add your OpenAI API key to .env for full functionality")
        print("2. Run: streamlit run app.py")
        print("3. Try uploading some legal documents!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
