from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class EmbeddingStore:
    def __init__(self, collection_name: str = "documents"):
        # Initialize embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize Chroma (local, in-memory)
        # self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(self, chunks: List[Dict], doc_id_prefix: str = "doc"):
        """Embed and store chunks into ChromaDB"""
        texts = [chunk["content"] for chunk in chunks]
        ids = [f"{doc_id_prefix}_{i}" for i in range(len(chunks))]
        embeddings = self.model.encode(texts).tolist()

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=[chunk["metadata"] for chunk in chunks]
        )
        print(f"✅ Stored {len(chunks)} chunks in vector DB")

    def query(self, question: str, top_k: int = 3):
        query_embedding = self.model.encode([question]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        return results



if __name__ == "__main__":
    from parser import load_pdf, chunk_text

    # Step 1: Load & chunk PDF
    text = load_pdf("data/test.pdf")
    chunks = chunk_text(text)

    # Step 2: Initialize store
    store = EmbeddingStore()

    # Step 3: Add chunks
    store.add_chunks(chunks, doc_id_prefix="sample")

    # Step 4: Query
    question = "What is Gaussian Random Walk?"
    results = store.query(question, top_k=2)
    print("🔍 Query:", question)
    print("Results:", results)
