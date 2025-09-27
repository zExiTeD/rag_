import chromadb

CHROMA_COLLECTION_NAME = "main_collection"
CHROMA_DB_PATH = "./chroma_db"

def client():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client

def get_collection( client ):
    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    except:
        collection = client.create_collection(name=CHROMA_COLLECTION_NAME)

    return collection

def add(collection, chunks , embeddings ):
    collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            ids=[f"doc_{i}" for i in range(len(chunks))]
            )

