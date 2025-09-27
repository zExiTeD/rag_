from modules import db
from modules import embeddings as embedder

chromadb_client = db.client()
chromadb_collection = db.get_collection( chromadb_client )

embedding_model = embedder.get_embedding_model()

