# A function that saves text to a Qdrant vector database 
# based on https://docs.camel-ai.org/key_modules/storages#3-5-using-qdrantstorage
# and https://docs.camel-ai.org/key_modules/embeddings#3-3-using-sentencetransformerencoder
# and https://python-client.qdrant.tech/qdrant_client.qdrant_client
import os
from camel.storages import QdrantStorage
from camel.embeddings import SentenceTransformerEncoder
from camel.storages import VectorRecord

class QdrantDB:
    """Simple helper class for Qdrant vector database operations."""
    
    def __init__(self, model_name: str = "TencentBAC/Conan-embedding-v1"):
        """Initialize the Qdrant database helper.
        Stages:
        1. Set the data storage root path
        2. Initialize the embedding model
        3. Create the QdrantStorage instance

        Params:
            model_name: the Hugging Face model name for embeddings
        """
        # Set the root path for data storage (use the current file's directory)
        rootpath = os.path.dirname(__file__)  # referring to the tutorial

        # Initialize the SentenceTransformerEncoder using the provided model name
        self.embedding_instance = SentenceTransformerEncoder(model_name=model_name)

        # Initialize the QdrantStorage with vector dimension and storage path
        vector_dim = self.embedding_instance.get_output_dim()  # embedding output dimension
        self.storage_instance = QdrantStorage(
            vector_dim=vector_dim, path=rootpath, collection_name='collection0'
        )
        
    def save_text(self, text: str, source_file: str = "unknown"):
        """Save a single text item into the Qdrant database.
        Stages:
        1. Convert the text to an embedding vector
        2. Create a VectorRecord with payload metadata
        3. Save the record into the storage

        Params:
            text: the text to save
            source_file: the source filename where the text came from
        """
        # Convert the text to an embedding vector using embed_list (expects a list)
        # Take a list of texts and return a list of vectors correspondingly
        vector = self.embedding_instance.embed_list([text])[0]

        # Create the payload dictionary that contains the text and source file info
        payload = {
            "text": text,
            "content path": source_file,
        }  # the file name is stored under 'content path' in VectorRetriever results

        # Create a VectorRecord with the vector and payload
        record = VectorRecord(
            vector=vector,  # the embedded vector representation of the text
            payload=payload,  # associated metadata
        )

        # Save the record into the Qdrant storage
        self.storage_instance.add([record])


if __name__ == "__main__":
    db = QdrantDB()
    db.save_text("这是第一段文本", "文档1.txt")
    db.save_text("这是第二段文本", "文档2.txt")
    print("文本保存完成！")
    # For debugging
    print(db.storage_instance.client.get_collection(collection_name='collection0'))
    db.storage_instance.client.delete_collection(collection_name='collection0') # not working