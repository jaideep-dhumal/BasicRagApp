import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import gradio as gr

# Global variables for Gradio
embed_model = None
rag_pipeline = None
chunks = None
chunk_embs = None

def load_docs(folder):
    docs = []
    for fname in os.listdir(folder):
        if fname.endswith('.txt'):
            with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
                docs.append(f.read())
    return docs

def chunk_text(text, chunk_size=500):
    # Simple chunking by sentences
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    chunk = ''
    for sent in sentences:
        if len(chunk) + len(sent) < chunk_size:
            chunk += ' ' + sent
        else:
            chunks.append(chunk.strip())
            chunk = sent
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def build_corpus(folder):
    docs = load_docs(folder)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))
    return all_chunks

def embed_corpus(chunks, model):
    return model.encode(chunks)

def retrieve(query, chunks, chunk_embs, embed_model, top_k=1):
    q_emb = embed_model.encode([query])
    sims = cosine_similarity(q_emb, chunk_embs)[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_idx]

def answer_question(query):
    context = retrieve(query, chunks, chunk_embs, embed_model)[0]
    prompt = f"Answer the question based on the context.\nContext: {context}\nQuestion: {query}"
    answer = rag_pipeline(prompt)[0]['generated_text'].strip()
    return answer

def main():
    folder = 'docs'
    print('Loading embedding model...')
    global embed_model, rag_pipeline, chunks, chunk_embs
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    print('Loading LLM...')
    rag_pipeline = pipeline('text2text-generation', model='google/flan-t5-small', max_new_tokens=128)
    print('Building corpus...')
    chunks = build_corpus(folder)
    chunk_embs = embed_corpus(chunks, embed_model)
    print('Launching Gradio chat app...')

    def chat_fn(message, history):
        answer = answer_question(message)
        return answer

    chat_iface = gr.ChatInterface(
        fn=chat_fn,
        title="RAG Q&A Chatbot",
        description="Ask a question about the documents in the docs folder.",
        examples=["Tell me about Earth's atmosphere", "What percent of Earth is water?"],
    )
    chat_iface.launch(share=False, inbrowser=True)

if __name__ == '__main__':
    main()
