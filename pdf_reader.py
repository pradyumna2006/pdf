import fitz  # PyMuPDF
from typing import List

def extract_and_chunk_pdf(pdf_file) -> List[str]:
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    
    # Chunk by paragraph
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    
    # Split long chunks further
    max_length = 500
    final_chunks = []
    for chunk in chunks:
        while len(chunk) > max_length:
            final_chunks.append(chunk[:max_length])
            chunk = chunk[max_length:]
        final_chunks.append(chunk)
    
    return final_chunks
