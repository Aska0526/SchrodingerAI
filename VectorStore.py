import chromadb
from sentence_transformers import SentenceTransformer

class VectorStore:

    def __init__(self, collection_name):
       # Initialize the embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

    # Method to populate the vector store with embeddings from a dataset
    def populate_vectors(self, dataset):
        for i, item in enumerate(dataset):
            combined_text = f"{item['message_1']}. {item['message_2']}"
            embeddings = self.embedding_model.encode(combined_text).tolist()
            self.collection.add(embeddings=[embeddings], documents=[item['message_2']], ids=[f"id_{i}"])

    # Method to search the ChromaDB collection for relevant context based on a query
    def search_context(self, query, n_results=1):
        query_embeddings = self.embedding_model.encode(query).tolist()
        return self.collection.query(query_embeddings=query_embeddings, n_results=n_results)