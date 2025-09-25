from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from embeddings import EmbeddingStore

class RAGPipeline:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        # Load embedding store
        self.store = EmbeddingStore()

        # Load LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.qa_pipeline = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)

    def ask(self, question: str, top_k: int = 3) -> str:
        # Retrieve top chunks from embeddings
        results = self.store.query(question, top_k=top_k)
        retrieved_chunks = results["documents"][0]

        if not retrieved_chunks:
            return "⚠️ No relevant context found in the database."

        # Build context
        context = " ".join(retrieved_chunks)

        # Prompt LLM
        prompt = f"Answer the following question using the context below:\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"

        # Generate answer
        answer = self.qa_pipeline(prompt, max_length=200, clean_up_tokenization_spaces=True)[0]["generated_text"]

        return answer


if __name__ == "__main__":
    rag = RAGPipeline()
    question = "Which university is this document written for?"
    answer = rag.ask(question)
    print("Q:", question)
    print("A:", answer)
