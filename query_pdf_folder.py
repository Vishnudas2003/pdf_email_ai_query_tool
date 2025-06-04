import os
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

PDF_FOLDER = os.path.expanduser("~/Desktop/TESTING")
CHUNK_SIZE = 1000

app = Flask(__name__)

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
                    "chunk_text": chunk,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
    return all_chunks, metadata

def get_embeddings(texts):
    return model.encode(texts, convert_to_numpy=True).astype("float32")

def extract_matching_sentence(chunk, keyword):
    sentences = re.split(r'(?<=[.!?])\s+', chunk)
    for sentence in sentences:
        if keyword.lower() in sentence.lower():
            return sentence.strip()
    return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query", "").strip()
    keywords = [kw.strip() for kw in query.split() if kw.strip()]
    
    if not keywords:
        return jsonify([])

    results = []
    seen = set()  # Avoid duplicates

    for keyword in keywords:
        query_embedding = model.encode([keyword])[0].astype("float32")
        k = 10
        distances, indices = index.search(np.array([query_embedding]), k)

        for idx in indices[0]:
            meta = metadata[idx]
            sentence = extract_matching_sentence(meta['chunk_text'], keyword)
            if sentence:
                result_key = (meta['file'], sentence)
                if result_key not in seen:
                    seen.add(result_key)
                    results.append({
                        "file": meta['file'],
                        "sentence": sentence,
                        "timestamp": meta['timestamp']
                    })

    return jsonify(results)

# Only run index building once
print("📁 Scanning PDF folder...")
chunks, metadata = extract_text_from_pdfs(PDF_FOLDER)
print(f"✅ Extracted {len(chunks)} chunks.")

print("🧠 Creating embeddings locally...")
embeddings = get_embeddings(chunks)

print("⚡ Building vector index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print("✅ Vector index built.")

if __name__ == "__main__":
    app.run(debug=True)
