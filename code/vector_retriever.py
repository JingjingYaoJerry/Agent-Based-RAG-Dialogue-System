# Retrieve related texts from the Qdrant vector database
# based on https://docs.camel-ai.org/key_modules/retrievers
from camel.retrievers import VectorRetriever
from qdrant import QdrantDB

class VecRetriever:
    """A simple vector retriever."""
    
    def __init__(self, qdrant_db: QdrantDB):
        """Initialize the vector retriever.
        Stages:
        1. Store the qdrant_db instance
        2. Create a VectorRetriever object

        Params:
            qdrant_db: QdrantDB instance (from qdrant.py)
        """
        # Store the qdrant_db instance
        self.qdrant_db = qdrant_db  # as instructed

        # Create VectorRetriever instance with QdrantDB's embedding_model and storage
        self.vector_retriever = VectorRetriever(
            embedding_model=qdrant_db.embedding_instance, # for data embedding
            storage=qdrant_db.storage_instance # for vector storage
        ) # using VectorRetriever considering the calling of "query" method as instructed
    
    def search(self, question: str, top_k: int = 5):
        """Retrieve related texts for a given question.
        Stages:
        1. Use vector_retriever.query() to retrieve results
        2. Format the returned results

        Params:
            question: the query string
            top_k: number of results to return

        Returns:
            A list where each element is {'file_name': 'filename', 'content': 'text'}
        """
        # Use self.vector_retriever.query() to perform the top-k retrieval
        results = self.vector_retriever.query(
            query=question,
            top_k=top_k
        )
        # Format the retrieval results into the specified structure
        # Original result format: [{'content path': 'filename', 'text': 'content'}, ...]
        # Target format: [{'file_name': 'filename', 'content': 'content'}, ...]
        # Re-assign the specified keys from the original results
        # ('similarity score' and 'metadata' are ignored)
        for item in results:
            item['file_name'] = item.pop('content path')
            item['content'] = item.pop('text')
        result = results
        return result


if __name__ == "__main__":
    # Initialize Qdrant database and save sample texts
    db = QdrantDB()
    db.save_text("人工智能是计算机科学的一个分支", "AI教程.txt")
    db.save_text("机器学习是实现人工智能的方法之一", "ML指南.txt")
    # Initialize the vector retriever
    retriever = VecRetriever(db)
    # Search for relevant texts
    results = retriever.search("什么是人工智能？", top_k=3)
    print(results)
    # Display results
    for item in results:
        print(f"文件：{item['file_name']}")
        print(f"内容：{item['content']}")
        print("-" * 50)