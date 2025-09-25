from embeddings import EmbeddingStore  # your working embeddings.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# --------------------------
# CONFIG
# --------------------------
collection_name = "documents"
top_k = 3
model_name = "Qwen/Qwen3Guard-Gen-8B"
max_new_tokens = 200

# --------------------------
# STEP 1: Initialize Embedding Store
# --------------------------
store = EmbeddingStore(collection_name=collection_name)

# --------------------------
# STEP 2: Initialize LLM (8-bit for small memory)
# --------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load in 8-bit if bitsandbytes is available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True,
    torch_dtype=torch.float16,
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # Use GPU if available
)

# --------------------------
# STEP 3: Define RAG function
# --------------------------
def answer_question(question: str) -> str:
    # Retrieve top-k chunks
    results = store.query(question, top_k=top_k)
    retrieved_chunks = results["documents"][0]

    if not retrieved_chunks:
        return "No relevant information found."

    context = "\n".join(retrieved_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7
    )[0]["generated_text"]

    # Remove the prompt to keep only the answer
    answer = output.split("Answer:")[-1].strip()
    return answer

# --------------------------
# STEP 4: Interactive testing
# --------------------------
if __name__ == "__main__":
    print("RAG pipeline ready. Type 'exit' to quit.")
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == "exit":
            break
        answer = answer_question(question)
        print("\n🔹 Answer:\n", answer)
