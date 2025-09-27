#!/usr/bin/env python3
"""
Setup script for RAG System
"""

import os
import json
import subprocess
import sys

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

def create_config():
    """Create default configuration if it doesn't exist"""
    config_path = "config.json"
    if not os.path.exists(config_path):
        print("⚙️ Creating default configuration...")
        default_config = {
            "llm": {
                "type": "ollama",
                "model_name": "llama2",
                "base_url": "http://localhost:11434"
            },
            "embedding": {
                "model": "all-mpnet-base-v2"
            },
            "audio": {
                "model": "base"
            },
            "database": {
                "path": "./chroma_db",
                "collection_name": "main_collection"
            },
            "processing": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "max_results": 5
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"✅ Configuration created: {config_path}")
    else:
        print(f"✅ Configuration already exists: {config_path}")

def create_directories():
    """Create necessary directories"""
    directories = ["chroma_db", "data"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory exists: {directory}")

def check_ollama():
    """Check if Ollama is available"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
    except:
        pass
    
    print("⚠️ Ollama not detected. To use local models:")
    print("  1. Install Ollama: https://ollama.ai/")
    print("  2. Start Ollama: ollama serve")
    print("  3. Pull a model: ollama pull llama2")
    return False

def main():
    """Main setup function"""
    print("🚀 RAG System Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Install dependencies
    install_dependencies()
    
    # Create configuration
    create_config()
    
    # Check Ollama
    check_ollama()
    
    print("\n✅ Setup completed!")
    print("\nNext steps:")
    print("1. Start Ollama (if using local models): ollama serve")
    print("2. Pull a model: ollama pull llama2")
    print("3. Run the example: python example.py")
    print("4. Start interactive mode: python main.py --interactive")

if __name__ == "__main__":
    main()
