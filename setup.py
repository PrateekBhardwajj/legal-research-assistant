"""
Setup script for the Legal Research Assistant.
Installs dependencies and initializes the system.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("Installing Python dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    return True

def setup_directories():
    """Create necessary directories."""
    print("Creating directory structure...")
    
    directories = [
        "logs",
        "chromadb",
        "temp",
        "data/contracts",
        "data/case_law", 
        "data/statutes",
        "src/__pycache__"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created!")

def create_env_file():
    """Create .env file if it doesn't exist."""
    if not Path(".env").exists():
        print("Creating .env file...")
        
        env_content = """# API Keys
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Application Settings
DEBUG=False
LOG_LEVEL=INFO

# Vector Database Settings
CHROMADB_HOST=localhost
CHROMADB_PORT=8000

# Security
SECRET_KEY=your_secret_key_here
"""
        
        with open(".env", "w") as f:
            f.write(env_content)
        
        print("✅ .env file created! Please update with your API keys.")
    else:
        print("✅ .env file already exists!")

def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded!")
    except ImportError:
        print("⚠️ NLTK not installed yet, will download data on first run.")

def create_init_files():
    """Create __init__.py files for Python modules."""
    print("Creating module initialization files...")
    
    init_files = [
        "src/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch(exist_ok=True)
    
    print("✅ Module files created!")

def main():
    """Main setup function."""
    print("🚀 Setting up Legal Research Assistant...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    
    # Run setup steps
    setup_directories()
    create_init_files()
    create_env_file()
    
    # Install dependencies
    if not install_requirements():
        return False
    
    download_nltk_data()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Update the .env file with your API keys")
    print("2. Run: streamlit run app.py")
    print("3. Upload some legal documents to test the system")
    
    return True

if __name__ == "__main__":
    main()
