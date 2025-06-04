import os
import re
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

PDF_FOLDER = os.path.expanduser("~/Desktop/TESTING")
CHUNK_SIZE = 1000

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdfs(folder_path):
    all_chunks = []
    metadata = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            reader = PdfReader(pdf_path)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() or ""
            for i in range(0, len(full_text), CHUNK_SIZE):
                chunk = full_text[i:i + CHUNK_SIZE]
                all_chunks.append(chunk)
                metadata.append({
                    "file": file,
                    "chunk_index": i // CHUNK_SIZE,
                    "chunk_text": chunk
                })
    return all_chunks, metadata

def get_embeddings(texts):
    return model.encode(texts, convert_to_numpy=True).astype("float32")

def extract_matching_sentence(chunk, keyword):
    # Split into rough sentences and find the one containing the keyword
    sentences = re.split(r'(?<=[.!?])\s+', chunk)
    for sentence in sentences:
        if keyword.lower() in sentence.lower():
            return sentence.strip()
    return None

def search_keyword(query, chunks, metadata, index):
    query_embedding = model.encode([query])[0].astype("float32")
    k = 10
    distances, indices = index.search(np.array([query_embedding]), k)

    matched_files = set()
    print(f"\n🔍 Results for: '{query}'")
    for idx in indices[0]:
        file = metadata[idx]['file']
        chunk_text = metadata[idx]['chunk_text']
        sentence = extract_matching_sentence(chunk_text, query)
        if sentence:
            matched_files.add(file)
            print(f"✅ Found in {file} (chunk {metadata[idx]['chunk_index']})")
            print(f"   📝 Matched Sentence: {sentence}")
    if not matched_files:
        print("❌ No direct match found, but these are similar based on context:")
        for idx in indices[0]:
            file = metadata[idx]['file']
            print(f"🔹 Possible match in {file} (chunk {metadata[idx]['chunk_index']})")

if __name__ == "__main__":
    print("📁 Scanning PDF folder...")
    chunks, metadata = extract_text_from_pdfs(PDF_FOLDER)
    print(f"✅ Extracted {len(chunks)} chunks.")

    print("🧠 Creating embeddings locally...")
    embeddings = get_embeddings(chunks)

    print("⚡ Building vector index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    while True:
        user_query = input("\nAsk your keyword (or type 'exit'): ").strip()
        if user_query.lower() == 'exit':
            break
        search_keyword(user_query, chunks, metadata, index)
