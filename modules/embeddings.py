from sentence_transformers import SentenceTransformer
import db

EMBEDDING_MODEL = "all-mpnet-base-v2"

def get_embedding_model() :
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return embedding_model

def embeddfile( data , embedding_model ):
    embeddings = embedding_model.encode(data, show_progress_bar=True)
    return embeddings

def add_to_db( embeddings ):
    db.add(collection, chunks , embeddings )


