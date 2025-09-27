from . import db
from . import embeddings
from . import parsing
from . import models
import whisper
import os
from typing import List, Dict, Any, Optional
import json


class RAGCore:
    """
    Main RAG (Retrieval-Augmented Generation) core class.
    Handles document embedding, storage, and querying with LLM integration.
    """
    
    def __init__(self, audio_model_name: str = "base", llm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize RAG core with database, models, and LLM.
        
        Args:
            audio_model_name: Whisper model size for audio transcription
            llm_config: LLM configuration dictionary
        """
        # Database setup
        self.database_client = db.client()
        self.database_collection = db.get_collection(self.database_client)
        
        # Model setup
        self.audio_model = whisper.load_model(audio_model_name)
        self.embedding_model = embeddings.get_embedding_model()
        
        # LLM setup
        if llm_config is None:
            llm_config = models.load_llm_config()
        self.llm = models.LLMFactory.create_llm(llm_config)
        
        # Configuration
        self.audio_model_name = audio_model_name
        self.llm_config = llm_config
    
    def embed_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
        """
        Extract content from file and embed it into the database.
        
        Args:
            file_path: Path to the file to process
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
            
        Returns:
            Dictionary with processing results
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"Processing file: {file_path}")
        
        # Extract content from file
        try:
            data = parsing.extract_content(file_path, self.audio_model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to extract content from {file_path}: {e}")
        
        # Process text content
        text_chunks = []
        if data.get("text"):
            # Create chunks from text content
            all_text = " ".join([item["content"] for item in data["text"]])
            chunks = self._create_chunks(all_text, chunk_size, chunk_overlap)
            text_chunks.extend(chunks)
        
        # Process images (store metadata for now)
        image_metadata = []
        if data.get("images"):
            for img in data["images"]:
                image_metadata.append({
                    "file": img["file"],
                    "page": img["page"],
                    "type": "image"
                })
        
        if not text_chunks:
            return {"status": "no_text", "chunks": 0, "images": len(image_metadata)}
        
        # Generate embeddings
        print(f"Generating embeddings for {len(text_chunks)} chunks...")
        try:
            embeddings_array = embeddings.embed_text(text_chunks, self.embedding_model)
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {e}")
        
        # Store in database
        try:
            embeddings.add_to_db(self.database_collection, text_chunks, embeddings_array)
            print(f"Successfully embedded {len(text_chunks)} chunks")
        except Exception as e:
            raise RuntimeError(f"Failed to store embeddings: {e}")
        
        return {
            "status": "success",
            "chunks": len(text_chunks),
            "images": len(image_metadata),
            "file": os.path.basename(file_path)
        }
    
    def _create_chunks(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Create overlapping text chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def query(self, question: str, max_results: int = 5, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            max_results: Maximum number of relevant documents to retrieve
            include_metadata: Whether to include document metadata
            
        Returns:
            Dictionary with query results and generated response
        """
        print(f"Querying: {question}")
        
        # Generate query embedding
        try:
            query_embedding = embeddings.embed_text([question], self.embedding_model)
        except Exception as e:
            raise RuntimeError(f"Failed to generate query embedding: {e}")
        
        # Search for relevant documents
        try:
            results = self.database_collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=max_results,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to query database: {e}")
        
        # Prepare context for LLM
        context_docs = results["documents"][0] if results["documents"] else []
        context = "\n\n".join(context_docs)
        
        # Generate response using LLM
        try:
            prompt = self._create_rag_prompt(question, context)
            response = self.llm.generate(prompt)
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {e}")
        
        # Prepare results
        result = {
            "question": question,
            "response": response,
            "context_documents": context_docs,
            "num_results": len(context_docs)
        }
        
        if include_metadata and results.get("metadatas"):
            result["metadata"] = results["metadatas"][0]
        
        return result
    
    def _create_rag_prompt(self, question: str, context: str) -> str:
        """Create a RAG prompt for the LLM"""
        return f"""Based on the following context, please answer the question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
    
    def chat(self, messages: List[Dict[str, str]], max_results: int = 5) -> str:
        """
        Chat interface with RAG system.
        
        Args:
            messages: List of chat messages
            max_results: Maximum number of relevant documents to retrieve
            
        Returns:
            Generated response
        """
        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        if not user_message:
            return "No user message found."
        
        # Query the RAG system
        try:
            result = self.query(user_message, max_results, include_metadata=False)
            context = "\n".join(result["context_documents"])
            
            # Add context to messages
            rag_messages = messages + [{"role": "system", "content": f"Context: {context}"}]
            
            # Generate response
            response = self.llm.chat(rag_messages)
            return response
            
        except Exception as e:
            return f"Error processing query: {e}"
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            count = self.database_collection.count()
            return {
                "collection_name": self.database_collection.name,
                "document_count": count,
                "embedding_model": "all-mpnet-base-v2",
                "llm_type": self.llm_config.get("type", "unknown")
            }
        except Exception as e:
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Delete and recreate collection
            self.database_client.delete_collection(self.database_collection.name)
            self.database_collection = db.get_collection(self.database_client)
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
    
    def update_llm_config(self, new_config: Dict[str, Any]):
        """Update LLM configuration and reload the model"""
        self.llm_config = new_config
        self.llm = models.LLMFactory.create_llm(new_config)
        models.save_llm_config(new_config)


# Backward compatibility
rag_core = RAGCore





