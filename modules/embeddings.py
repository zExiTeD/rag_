from sentence_transformers import SentenceTransformer
from . import db

EMBEDDING_MODEL = "all-mpnet-base-v2"

def get_embedding_model():
    """Get the embedding model instance"""
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return embedding_model

def embed_text(text_data, embedding_model):
    """Generate embeddings for text data"""
    if isinstance(text_data, str):
        text_data = [text_data]
    embeddings = embedding_model.encode(text_data, show_progress_bar=True)
    return embeddings

def add_to_db(collection, chunks, embeddings):
    """Add chunks and embeddings to the database"""
    db.add(collection, chunks, embeddings)


