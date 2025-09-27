#!/usr/bin/env python3
"""
Example usage of RAG system
"""

import os
from modules.core import RAGCore
from modules import models

def example_basic_usage():
    """Basic usage example"""
    print("🚀 RAG System Basic Usage Example")
    print("=" * 50)
    
    # Initialize RAG system with default config
    rag = RAGCore()
    
    # Example: Embed a text file (create one if it doesn't exist)
    sample_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create 
    intelligent machines that can perform tasks that typically require human intelligence. 
    These tasks include learning, reasoning, problem-solving, perception, and language understanding.
    
    Machine Learning is a subset of AI that focuses on algorithms that can learn and 
    make decisions from data. Deep Learning is a subset of machine learning that uses 
    neural networks with multiple layers to model and understand complex patterns.
    
    Natural Language Processing (NLP) is another important area of AI that focuses on 
    the interaction between computers and humans through natural language.
    """
    
    # Create sample file
    with open("sample_ai.txt", "w") as f:
        f.write(sample_text)
    
    try:
        # Embed the file
        print("📁 Embedding sample file...")
        result = rag.embed_file("sample_ai.txt")
        print(f"✅ Embedded {result['chunks']} chunks")
        
        # Query the system
        print("\n💬 Querying the system...")
        questions = [
            "What is Artificial Intelligence?",
            "What is the difference between AI and Machine Learning?",
            "What is Natural Language Processing?"
        ]
        
        for question in questions:
            print(f"\n❓ Question: {question}")
            response = rag.query(question)
            print(f"🤖 Answer: {response['response']}")
            print(f"📚 Found {response['num_results']} relevant documents")
        
        # Show collection info
        print("\n📊 Collection Information:")
        info = rag.get_collection_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Clean up
        if os.path.exists("sample_ai.txt"):
            os.remove("sample_ai.txt")


def example_different_llms():
    """Example showing different LLM configurations"""
    print("\n🔧 Different LLM Configuration Examples")
    print("=" * 50)
    
    # Example configurations
    configs = {
        "ollama": {
            "type": "ollama",
            "model_name": "llama2",
            "base_url": "http://localhost:11434"
        },
        "transformers": {
            "type": "transformers",
            "model_name": "microsoft/DialoGPT-medium",
            "device": "auto"
        },
        "openai": {
            "type": "openai",
            "model_name": "gpt-3.5-turbo",
            "api_key": "your-api-key-here"
        }
    }
    
    for llm_type, config in configs.items():
        print(f"\n🔧 {llm_type.upper()} Configuration:")
        print(f"  Type: {config['type']}")
        print(f"  Model: {config['model_name']}")
        if 'base_url' in config:
            print(f"  Base URL: {config['base_url']}")
        if 'device' in config:
            print(f"  Device: {config['device']}")


def example_custom_config():
    """Example with custom configuration"""
    print("\n⚙️ Custom Configuration Example")
    print("=" * 50)
    
    # Custom configuration
    custom_config = {
        "llm": {
            "type": "ollama",
            "model_name": "llama2",
            "base_url": "http://localhost:11434"
        },
        "audio": {
            "model": "base"
        }
    }
    
    try:
        # Initialize with custom config
        rag = RAGCore(
            audio_model_name="base",
            llm_config=custom_config["llm"]
        )
        
        print("✅ RAG system initialized with custom configuration")
        print(f"  LLM Type: {custom_config['llm']['type']}")
        print(f"  LLM Model: {custom_config['llm']['model_name']}")
        print(f"  Audio Model: {custom_config['audio']['model']}")
        
    except Exception as e:
        print(f"❌ Error initializing with custom config: {e}")


def example_file_processing():
    """Example of processing different file types"""
    print("\n📄 File Processing Example")
    print("=" * 50)
    
    # Create sample files
    sample_files = {
        "sample.txt": "This is a sample text file for testing the RAG system.",
        "sample.docx": "This would be a Word document in a real scenario.",
        "sample.pdf": "This would be a PDF document in a real scenario."
    }
    
    # Create text file
    with open("sample.txt", "w") as f:
        f.write(sample_files["sample.txt"])
    
    try:
        rag = RAGCore()
        
        # Process the text file
        print("📁 Processing sample.txt...")
        result = rag.embed_file("sample.txt", chunk_size=500, chunk_overlap=100)
        print(f"✅ Processed: {result['chunks']} chunks")
        
        # Query about the file
        response = rag.query("What is in the sample file?")
        print(f"🤖 Response: {response['response']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Clean up
        if os.path.exists("sample.txt"):
            os.remove("sample.txt")


def main():
    """Run all examples"""
    print("🎯 RAG System Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_different_llms()
    example_custom_config()
    example_file_processing()
    
    print("\n✅ All examples completed!")
    print("\nTo run the interactive mode:")
    print("  python main.py --interactive")
    print("\nTo embed files and query:")
    print("  python main.py --embed file1.pdf file2.docx --query 'Your question here'")


if __name__ == "__main__":
    main()
