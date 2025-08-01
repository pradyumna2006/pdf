
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import os

embedder = SentenceTransformer('all-MiniLM-L6-v2')
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def get_answer(question, pdf_file):
    # Extract text from PDF and split into chunks (paragraphs)
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]


    def call_groq_api(prompt):
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "You are an academic assistant. Answer the user's question clearly and concisely."},
                {"role": "user", "content": prompt}
            ]
        }
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=20)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                return None
        except Exception:
            return None

    if not chunks:
        # If no PDF content, fallback to Groq API
        ai_response = call_groq_api(question)
        if ai_response:
            return ai_response
        return "Sorry, I couldn't find relevant information in the PDF to answer your question."

    # Advanced semantic search: use embeddings
    chunk_embeddings = embedder.encode(chunks)
    question_embedding = embedder.encode([question])[0]
    similarities = np.dot(chunk_embeddings, question_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(question_embedding) + 1e-8)
    best_idx = int(np.argmax(similarities))
    best_chunk = chunks[best_idx]

    # Extractive summarization: return the most relevant sentences from the best chunk and its neighbors
    sentences = [s.strip() for s in best_chunk.replace('\n', ' ').split('. ') if s.strip()]
    sent_scores = [sum(1 for word in question.lower().split() if word in s.lower()) for s in sentences]
    top_indices = np.argsort(sent_scores)[::-1]
    top_sentences = [sentences[i] for i in top_indices if sent_scores[i] > 0]

    # Also consider context from neighboring chunks for richer answers
    context_chunks = [best_chunk]
    if best_idx > 0:
        context_chunks.insert(0, chunks[best_idx-1])
    if best_idx < len(chunks)-1:
        context_chunks.append(chunks[best_idx+1])
    context_text = ' '.join(context_chunks)
    context_sentences = [s.strip() for s in context_text.replace('\n', ' ').split('. ') if s.strip()]
    context_sent_scores = [sum(1 for word in question.lower().split() if word in s.lower()) for s in context_sentences]
    context_top_indices = np.argsort(context_sent_scores)[::-1]
    context_top_sentences = [context_sentences[i] for i in context_top_indices if context_sent_scores[i] > 0][:4]

    def is_definition_question(q):
        ql = q.lower()
        return ql.startswith('what is') or ql.startswith('who is') or ql.startswith('define') or ql.startswith('explain')

    if not context_top_sentences:
        # If no relevant PDF content, fallback to Groq API
        ai_response = call_groq_api(question)
        if ai_response:
            return ai_response
        return "Sorry, I couldn't find relevant information in the PDF to answer your question."
    else:
        if is_definition_question(question):
            entity = question.split(' ', 2)[-1].replace('?', '').strip()
            # Aggregate all sentences mentioning the entity
            relevant = [s for s in context_top_sentences if entity.lower() in s.lower()]
            # If not enough, add more top sentences
            if len(relevant) < 2:
                relevant += [s for s in context_top_sentences if s not in relevant][:2-len(relevant)]
            # Synthesize a multi-sentence definition with rephrasing
            if relevant:
                # Try Groq API for a better definition
                prompt = f"{question}\n\nContext from PDF:\n" + " ".join(relevant)
                ai_response = call_groq_api(prompt)
                if ai_response:
                    return ai_response
                answer = f"{entity.capitalize()} can be described as follows:\n"
                for i, s in enumerate(relevant, 1):
                    if entity.lower() in s.lower():
                        s = s.replace(entity, entity.capitalize())
                        s = s.replace(entity.capitalize(), f"**{entity.capitalize()}**")
                    answer += f"{i}. {s.strip('. ')}.\n"
                answer += f"\nIn summary, {entity} is an important concept discussed in your PDF."
                return answer.strip()
            else:
                ai_response = call_groq_api(question)
                if ai_response:
                    return ai_response
                return f"Sorry, I couldn't find a clear definition for '{entity}' in the PDF."
        else:
            # General answer: aggregate and rephrase, or use Groq API
            prompt = f"{question}\n\nContext from PDF:\n" + " ".join(context_top_sentences[:3])
            ai_response = call_groq_api(prompt)
            if ai_response:
                return ai_response
            answer = " ".join(context_top_sentences[:3])
            return f"**Answer:** {answer}\n\n_Context found in PDF for your question: '{question}'_"
