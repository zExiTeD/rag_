import db
import embeddings
import parsing
import models

class rag_core:
    def __init__( self ):
        self.data_base_client = db.client()
        self.data_base_collection = db.get_collection( self.data_base_client )
        self.audio_model = 
        self.embeddin_model = embeddings.get_embedding_model()
        self.llm = None

    def embedd(self , file_path ):
        data = parsing.extract_content( file_path )





