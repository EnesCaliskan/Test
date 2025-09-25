from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page_num, page in enumerate(reader.pages):
        text += page.extract_text() or ""
    return text

def load_txt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8")

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return [
        {"content": chunk, "metadata": {"length": len(chunk)}}
        for chunk in chunks
    ]

if __name__ == "__main__":
   pdf_path = "data/test.pdf"
   text = load_pdf(pdf_path)
   print("Extracted text:\n", text[:300], "...\n")

   chunks = chunk_text(text)
   print(f"Created {len(chunks)} chunks")
   print("First chunk:", chunks[0])

