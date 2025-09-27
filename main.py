#!/usr/bin/env python3
"""
RAG System - Smart Multi-modal Information Handler
A RAG (Retrieval-Augmented Generation) system for processing and querying documents.
"""

import os
import json
import argparse
from typing import List, Dict, Any
from modules.core import RAGCore
from modules import models


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Return default configuration
        return {
            "llm": {"type": "ollama", "model_name": "llama2", "base_url": "http://localhost:11434"},
            "embedding": {"model": "all-mpnet-base-v2"},
            "audio": {"model": "base"},
            "database": {"path": "./chroma_db", "collection_name": "main_collection"},
            "processing": {"chunk_size": 1000, "chunk_overlap": 200, "max_results": 5}
        }


def save_config(config: Dict[str, Any], config_path: str = "config.json"):
    """Save configuration to file"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def initialize_rag(config: Dict[str, Any]) -> RAGCore:
    """Initialize RAG system with configuration"""
    llm_config = config.get("llm", {})
    audio_model = config.get("audio", {}).get("model", "base")
    
    return RAGCore(
        audio_model_name=audio_model,
        llm_config=llm_config
    )


def embed_files(rag: RAGCore, file_paths: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Embed multiple files into the RAG system"""
    results = {}
    chunk_size = config.get("processing", {}).get("chunk_size", 1000)
    chunk_overlap = config.get("processing", {}).get("chunk_overlap", 200)
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            results[file_path] = {"error": "File not found"}
            continue
        
        try:
            result = rag.embed_file(file_path, chunk_size, chunk_overlap)
            results[file_path] = result
            print(f"✅ Processed {file_path}: {result['chunks']} chunks")
        except Exception as e:
            results[file_path] = {"error": str(e)}
            print(f"❌ Failed to process {file_path}: {e}")
    
    return results


def query_rag(rag: RAGCore, question: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Query the RAG system"""
    max_results = config.get("processing", {}).get("max_results", 5)
    
    try:
        result = rag.query(question, max_results)
        return result
    except Exception as e:
        return {"error": str(e)}


def interactive_mode(rag: RAGCore, config: Dict[str, Any]):
    """Interactive query mode"""
    print("\n🤖 RAG System Interactive Mode")
    print("Type 'quit' to exit, 'info' for collection info, 'clear' to clear collection")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n💬 Your question: ").strip()
            
            if question.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif question.lower() == 'info':
                info = rag.get_collection_info()
                print(f"📊 Collection Info: {info}")
                continue
            elif question.lower() == 'clear':
                if rag.clear_collection():
                    print("🗑️ Collection cleared!")
                else:
                    print("❌ Failed to clear collection")
                continue
            elif not question:
                continue
            
            result = query_rag(rag, question, config)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"\n🤖 Answer: {result['response']}")
                if result.get('context_documents'):
                    print(f"\n📚 Found {result['num_results']} relevant documents")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="RAG System - Smart Multi-modal Information Handler")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--embed", nargs="+", help="Files to embed into the system")
    parser.add_argument("--query", help="Single query to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive mode")
    parser.add_argument("--info", action="store_true", help="Show collection information")
    parser.add_argument("--clear", action="store_true", help="Clear the collection")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize RAG system
    try:
        rag = initialize_rag(config)
        print("✅ RAG system initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return
    
    # Handle different modes
    if args.clear:
        if rag.clear_collection():
            print("🗑️ Collection cleared!")
        else:
            print("❌ Failed to clear collection")
        return
    
    if args.info:
        info = rag.get_collection_info()
        print(f"📊 Collection Info: {json.dumps(info, indent=2)}")
        return
    
    if args.embed:
        print(f"📁 Embedding {len(args.embed)} files...")
        results = embed_files(rag, args.embed, config)
        
        # Print summary
        successful = sum(1 for r in results.values() if "error" not in r)
        failed = len(results) - successful
        print(f"\n📊 Summary: {successful} successful, {failed} failed")
        
        if args.query:
            # Process query after embedding
            result = query_rag(rag, args.query, config)
            if "error" in result:
                print(f"❌ Query error: {result['error']}")
            else:
                print(f"\n🤖 Answer: {result['response']}")
    
    elif args.query:
        result = query_rag(rag, args.query, config)
        if "error" in result:
            print(f"❌ Query error: {result['error']}")
        else:
            print(f"🤖 Answer: {result['response']}")
    
    elif args.interactive:
        interactive_mode(rag, config)
    
    else:
        # Default: show help
        parser.print_help()


if __name__ == "__main__":
    main()

