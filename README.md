# ⚖️ CyberlawGPT

## Pakistan Cyber Law RAG Assistant

CyberlawGPT is a Retrieval-Augmented Generation (RAG) application
that answers questions about Pakistani cyber laws using official
legal documents and the Groq API.

The application uses:

- Python
- Streamlit
- FAISS
- Sentence Transformers
- PyMuPDF
- Groq
- NCCIA official cyber-law sources

---

# Features

## 1. Automatic legal-document download

When the application starts, it attempts to access:

NCCIA Cyber Laws

https://nccia.gov.pk/laws.php

The application searches the official NCCIA cyber-law page for
available PDF documents.

If the NCCIA page cannot be accessed automatically, the application
uses official government fallback copies of relevant legislation.

---

# 2. Automatic embeddings

Every startup performs the following process:

NCCIA Cyber Laws
        ↓
Download PDF
        ↓
Extract PDF text
        ↓
Clean text
        ↓
Split into overlapping chunks
        ↓
Sentence Transformer embeddings
        ↓
FAISS index
        ↓
Ready for questions

No manually prepared vector database is required.

---

# 3. RAG question answering

When a user asks a question:

User Question
      ↓
Sentence Transformer embedding
      ↓
FAISS similarity search
      ↓
Relevant legal provisions
      ↓
Groq LLM
      ↓
Grounded answer

The model is instructed not to invent:

- sections
- penalties
- offences
- authorities
- procedures
- definitions
- legal requirements

---

# 4. Technicality levels

The UI provides:

### Beginner

Simple explanations with minimal legal terminology.

### Intermediate

Balanced explanation with important legal terminology.

### Expert

More precise legal terminology and section-level discussion.

---

# 5. Response size

Available options:

- Short
- Medium
- Detailed
- Very Detailed

---

# 6. Languages

CyberlawGPT supports:

- English
- Urdu
- Roman Urdu
- English + Urdu

---

# 7. FAISS controls

The sidebar allows the user to adjust:

### Retrieved legal passages

Controls how many relevant chunks are supplied to Groq.

### Chunk size

Controls the approximate size of each legal-text chunk.

### Chunk overlap

Provides overlapping text between neighboring chunks.

These controls are useful for experimenting with retrieval quality.

---

# 8. Groq models

The UI currently provides:

- openai/gpt-oss-20b
- openai/gpt-oss-120b

The application uses the Groq API for generation.

---

# Requirements

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
