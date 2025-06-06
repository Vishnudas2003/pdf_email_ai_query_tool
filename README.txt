🔍 PDF Keyword Search Tool (Offline + Local Embeddings)

This tool allows you to search across multiple PDFs using a keyword, and returns:

📄 Which PDF file contains the keyword
✂️ Exact sentence where the keyword appears
It uses local embeddings with Sentence Transformers + FAISS, ensuring:

✅ No API keys required
✅ Works offline
✅ Scales for large document sets
📁 Folder Setup
Place all your PDF files inside a folder.
Example path (default):

~/Desktop/TESTING
🛠️ Installation
# (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install PyPDF2 faiss-cpu sentence-transformers
▶️ Running the Tool
python3 search_pdf_keywords.py
Then type a keyword like:

money
You'll see:

✅ Found in conversation1.pdf (chunk 0)
   📝 Matched Sentence: We discussed the money situation last week.
🧠 How It Works
📄 Reads all PDFs and splits their text into chunks.
🔎 Creates local vector embeddings using sentence-transformers.
⚡ Builds a FAISS index for fast keyword similarity search.
💬 When you enter a keyword, it:
Finds the best-matching chunks
Scans for exact sentence with the keyword
Shows matching PDF and sentence
✅ Features
Works offline (no internet or API required)
Lightweight and fast
Easily scalable
Useful for searching through email-based PDFs, meeting notes, etc.
🗃️ Example Use Cases
Finding specific topics or discussions across thousands of internal PDF documents
Quickly identifying which documents reference money, invoice, client names, or any keyword

this is a demo
