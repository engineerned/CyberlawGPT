import io
import os
import re
import hashlib
from typing import List, Dict, Tuple

import requests
import fitz  # PyMuPDF
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# CYBERLAWGPT
# Pakistan Cyber Law RAG Assistant
# Source: National Cyber Crime Investigation Agency (NCCIA)
# ============================================================

APP_NAME = "CyberlawGPT"

NCCIA_LAWS_URL = "https://nccia.gov.pk/laws.php"

OFFICIAL_SOURCES = [
    {
        "name": "Prevention of Electronic Crimes Act 2016",
        "url": (
            "https://www.pakistancode.gov.pk/"
            "pdffiles/administrator6a061efe0ed5bd153fa8b79b8eb4cba7.pdf"
        ),
    },
    {
        "name": "Prevention of Electronic Crimes Amendment Act 2025",
        "url": (
            "https://senate.gov.pk/uploads/documents/"
            "1737697316_896.pdf"
        ),
    },
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CyberlawGPT",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 0;
        }

        .subtitle {
            font-size: 17px;
            color: #6b7280;
            margin-top: 0;
            margin-bottom: 25px;
        }

        .law-card {
            padding: 18px;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,.25);
            margin-bottom: 12px;
        }

        .source-box {
            padding: 12px;
            border-left: 4px solid #4f46e5;
            background: rgba(79,70,229,.06);
            border-radius: 8px;
            margin-top: 10px;
        }

        .warning-box {
            padding: 14px;
            border-radius: 10px;
            background: rgba(245,158,11,.10);
            border: 1px solid rgba(245,158,11,.35);
        }

        .success-box {
            padding: 14px;
            border-radius: 10px;
            background: rgba(16,185,129,.10);
            border: 1px solid rgba(16,185,129,.35);
        }

        .stChatMessage {
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# HELPERS
# ============================================================

def clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def download_pdf(url: str, timeout: int = 60) -> bytes:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/153.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "application/pdf,application/octet-stream,"
            "text/html;q=0.9,*/*;q=0.8"
        ),
        "Referer": NCCIA_LAWS_URL,
        "Connection": "keep-alive",
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=timeout,
        allow_redirects=True,
    )

    response.raise_for_status()

    data = response.content

    if data.startswith(b"%PDF"):
        return data

    # Some servers return PDF with unusual headers.
    content_type = response.headers.get(
        "content-type", ""
    ).lower()

    if "pdf" in content_type:
        return data

    raise ValueError(
        f"Expected PDF but received "
        f"{content_type or 'unknown content type'}"
    )


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from a PDF using PyMuPDF."""
    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []

    for page_number, page in enumerate(document):
        text = page.get_text("text")

        if text:
            pages.append(
                f"\n--- PAGE {page_number + 1} ---\n{text}"
            )

    document.close()

    return clean_text("\n".join(pages))


def extract_pdf_links_from_nccia(html: str) -> List[Dict[str, str]]:
    """
    Extract likely PDF links from NCCIA laws page.

    NCCIA can change its HTML structure, so this intentionally
    searches broadly for PDF links rather than relying on a
    particular CSS selector.
    """

    links = re.findall(
        r'href\s*=\s*["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']',
        html,
        flags=re.IGNORECASE,
    )

    results = []

    for link in links:
        if link.startswith("//"):
            link = "https:" + link

        elif link.startswith("/"):
            link = "https://nccia.gov.pk" + link

        elif not link.startswith("http"):
            link = "https://nccia.gov.pk/" + link

        results.append(
            {
                "name": os.path.basename(link.split("?")[0]),
                "url": link,
            }
        )

    return results


def discover_nccia_pdfs() -> List[Dict[str, str]]:
    """
    Discover PDF links from the official NCCIA cyber laws page.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/130 Safari/537.36"
        )
    }

    response = requests.get(
        NCCIA_LAWS_URL,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    links = extract_pdf_links_from_nccia(response.text)

    # Remove duplicates.
    unique = {}

    for item in links:
        unique[item["url"]] = item

    return list(unique.values())


def download_law_documents():
    """
    Download Pakistani cyber-law documents.

    Strategy:

    1. Try the NCCIA cyber-laws page.
    2. If NCCIA blocks automated access, do NOT fail.
    3. Use official government copies from Pakistan Code
       and Senate of Pakistan.
    """

    documents = []

    # --------------------------------------------------------
    # STEP 1 — Try NCCIA
    # --------------------------------------------------------

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/153.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
        }

        response = requests.get(
            NCCIA_LAWS_URL,
            headers=headers,
            timeout=30,
        )

        if response.status_code == 200:

            pdf_links = re.findall(
                r'href\s*=\s*["\']([^"\']+)["\']',
                response.text,
                flags=re.IGNORECASE,
            )

            for link in pdf_links:

                if ".pdf" not in link.lower():
                    continue

                if link.startswith("//"):
                    link = "https:" + link

                elif link.startswith("/"):
                    link = "https://nccia.gov.pk" + link

                elif not link.startswith("http"):
                    link = (
                        "https://nccia.gov.pk/"
                        + link
                    )

                try:

                    pdf_bytes = download_pdf(link)

                    text = extract_pdf_text(
                        pdf_bytes
                    )

                    if len(text) > 500:

                        documents.append(
                            {
                                "name": os.path.basename(
                                    link.split("?")[0]
                                ),
                                "url": link,
                                "text": text,
                            }
                        )

                except Exception:
                    continue

    except Exception:
        # NCCIA may return 403.
        # Continue to official government sources.
        pass

    # --------------------------------------------------------
    # STEP 2 — Official government sources
    # --------------------------------------------------------

    already_loaded = {
        x["url"]
        for x in documents
    }

    for source in OFFICIAL_SOURCES:

        if source["url"] in already_loaded:
            continue

        try:

            pdf_bytes = download_pdf(
                source["url"]
            )

            text = extract_pdf_text(
                pdf_bytes
            )

            if len(text) > 500:

                documents.append(
                    {
                        "name": source["name"],
                        "url": source["url"],
                        "text": text,
                    }
                )

        except Exception as error:

            print(
                f"Could not load "
                f"{source['name']}: {error}"
            )

    # --------------------------------------------------------
    # STEP 3 — Final validation
    # --------------------------------------------------------

    if not documents:

        raise RuntimeError(
            "No Pakistani cyber-law documents could be downloaded. "
            "NCCIA may be temporarily blocking automated requests "
            "or the official government sources may be unavailable."
        )

    return documents


def split_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> List[str]:
    """
    Split legal text into overlapping chunks.

    The splitter attempts to preserve sections and paragraphs.
    """

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:

        if len(current) + len(paragraph) + 2 <= chunk_size:
            current += (
                ("\n\n" if current else "")
                + paragraph
            )

        else:

            if current:
                chunks.append(current.strip())

            # Handle unusually large paragraphs.
            if len(paragraph) > chunk_size:

                start = 0

                while start < len(paragraph):

                    end = min(
                        start + chunk_size,
                        len(paragraph),
                    )

                    part = paragraph[start:end].strip()

                    if part:
                        chunks.append(part)

                    start = max(
                        end - overlap,
                        start + 1,
                    )

                current = ""

            else:
                current = paragraph

    if current:
        chunks.append(current.strip())

    return chunks


def extract_section_reference(text: str) -> str:
    """Try to identify a PECA section number from a chunk."""

    patterns = [
        r"\bsection\s+(\d+[A-Za-z]?)\b",
        r"\bSec\.\s*(\d+[A-Za-z]?)\b",
        r"^\s*(\d+[A-Za-z]?)\.\s",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return f"Section {match.group(1)}"

    return "Relevant provision"


def build_chunks(
    documents: List[Dict[str, str]],
    chunk_size: int,
    overlap: int,
) -> Tuple[List[str], List[Dict[str, str]]]:

    chunks = []
    metadata = []

    for document in documents:

        pieces = split_text(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, piece in enumerate(pieces):

            chunks.append(piece)

            metadata.append(
                {
                    "document": document["name"],
                    "url": document["url"],
                    "chunk": index + 1,
                    "section": extract_section_reference(piece),
                }
            )

    return chunks, metadata


# ============================================================
# EMBEDDING / FAISS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner=False)
def create_rag_index(
    chunk_size: int,
    overlap: int,
):

    documents = download_law_documents()

    if not documents:
        raise RuntimeError(
            "Unable to download cyber-law documents from "
            "NCCIA or the official government fallback sources."
        )

    chunks, metadata = build_chunks(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks:
        raise RuntimeError(
            "No usable legal text was extracted from the PDFs."
        )

    model = load_embedding_model()

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    # Build a source fingerprint.
    fingerprint_source = "".join(
        document["name"] + document["url"]
        for document in documents
    )

    fingerprint = hashlib.sha256(
        fingerprint_source.encode("utf-8")
    ).hexdigest()[:12]

    return {
        "index": index,
        "chunks": chunks,
        "metadata": metadata,
        "documents": documents,
        "fingerprint": fingerprint,
        "embedding_model": "all-MiniLM-L6-v2",
    }


def retrieve_documents(
    query: str,
    rag_data: Dict,
    top_k: int,
) -> List[Dict]:

    model = load_embedding_model()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = rag_data["index"].search(
        query_embedding,
        min(top_k, rag_data["index"].ntotal),
    )

    results = []

    for score, index in zip(scores[0], indices[0]):

        if index < 0:
            continue

        results.append(
            {
                "score": float(score),
                "text": rag_data["chunks"][index],
                "metadata": rag_data["metadata"][index],
            }
        )

    return results


# ============================================================
# GROQ
# ============================================================

def get_groq_key() -> str:

    # Streamlit Cloud:
    # st.secrets["GROQ_API_KEY"]

    try:
        secret_key = st.secrets.get("GROQ_API_KEY")

        if secret_key:
            return secret_key

    except Exception:
        pass

    # Optional local environment variable.
    return os.getenv("GROQ_API_KEY", "")


def create_groq_client():

    api_key = get_groq_key()

    if not api_key:
        return None

    return Groq(api_key=api_key)


def response_instruction(response_size: str) -> str:

    instructions = {
        "Short": "Give a concise answer, normally 2–4 paragraphs.",
        "Medium": "Give a balanced answer with useful explanation and legal references.",
        "Detailed": "Give a detailed answer with relevant sections, explanation, limitations and practical guidance.",
        "Very Detailed": "Give a comprehensive research-style answer, while staying strictly grounded in the retrieved law.",
    }

    return instructions.get(
        response_size,
        instructions["Medium"],
    )


def build_prompt(
    question: str,
    context: str,
    technicality: str,
    response_size: str,
    language: str,
) -> str:

    technicality_instruction = {
        "Beginner": (
            "Explain legal concepts in simple language. "
            "Avoid unnecessary legal jargon."
        ),
        "Intermediate": (
            "Use moderately technical legal terminology and "
            "explain important terms."
        ),
        "Expert": (
            "Use precise legal terminology and provide section-level "
            "analysis where the retrieved material supports it."
        ),
    }

    language_instruction = {
        "English": "Answer in English.",
        "Urdu": "Answer in Urdu using clear Pakistani Urdu.",
        "Roman Urdu": "Answer in Roman Urdu.",
        "English + Urdu": "Explain the answer in English and provide important points in Urdu.",
    }

    return f"""
You are CyberlawGPT, a Pakistan cyber-law information assistant.

Your primary legal source is the retrieved text from official Pakistani
cyber-law documents.

IMPORTANT RULES:

1. Answer the user's question using the retrieved legal context.
2. Do not invent a section, offence, penalty, authority, procedure,
   definition or legal requirement.
3. If the retrieved material does not contain enough information,
   clearly say that the available documents do not establish the answer.
4. When possible, identify the relevant Act and section.
5. Distinguish between:
   - what the law expressly states,
   - reasonable explanation of that text,
   - practical/general information.
6. Do not claim to be a lawyer.
7. Do not provide a definitive prediction about what a court or
   investigating agency will decide.
8. Do not encourage illegal activity.
9. If the question describes potentially criminal conduct, explain the
   relevant legal provisions without giving instructions for committing
   or evading a crime.
10. If the user asks for current procedural information, say when the
    retrieved documents may not establish the current procedure.
11. Do not fabricate citations.
12. If there is conflicting or incomplete material in the retrieved
    documents, explicitly mention the limitation.

Technicality:
{technicality_instruction[technicality]}

Response size:
{response_instruction(response_size)}

Language:
{language_instruction[language]}

RETRIEVED LEGAL CONTEXT
=======================

{context}

USER QUESTION
=============

{question}

RESPONSE FORMAT

Answer:
Provide the answer.

Relevant legal provisions:
List the relevant sections/documents if supported by the retrieved context.

Why they matter:
Briefly explain how those provisions relate to the question.

Important limitation:
Mention any uncertainty or missing information when appropriate.
"""


def ask_groq(
    question: str,
    retrieved: List[Dict],
    technicality: str,
    response_size: str,
    language: str,
    model_name: str,
) -> str:

    client = create_groq_client()

    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Add it to Streamlit Secrets or the GROQ_API_KEY "
            "environment variable."
        )

    context_parts = []

    for i, item in enumerate(retrieved, start=1):

        metadata = item["metadata"]

        context_parts.append(
            f"""
SOURCE {i}
Document: {metadata['document']}
Section indicator: {metadata['section']}
Similarity: {item['score']:.3f}

{item['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = build_prompt(
        question=question,
        context=context,
        technicality=technicality,
        response_size=response_size,
        language=language,
    )

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful Pakistan cyber-law information "
                    "assistant. Ground your answer in the supplied legal "
                    "context and never invent legal provisions."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.1,
        max_tokens=1800,
    )

    return completion.choices[0].message.content


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ CyberlawGPT Settings")

    st.subheader("Answer Controls")

    technicality = st.selectbox(
        "Technicality level",
        [
            "Beginner",
            "Intermediate",
            "Expert",
        ],
        index=1,
    )

    response_size = st.selectbox(
        "Response size",
        [
            "Short",
            "Medium",
            "Detailed",
            "Very Detailed",
        ],
        index=1,
    )

    language = st.selectbox(
        "Answer language",
        [
            "English",
            "Urdu",
            "Roman Urdu",
            "English + Urdu",
        ],
        index=0,
    )

    st.subheader("RAG Settings")

    top_k = st.slider(
        "Retrieved legal passages",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
    )

    chunk_size = st.slider(
        "Chunk size",
        min_value=600,
        max_value=1800,
        value=1000,
        step=100,
    )

    overlap = st.slider(
        "Chunk overlap",
        min_value=50,
        max_value=300,
        value=150,
        step=25,
    )

    st.subheader("Groq Model")

    groq_model = st.selectbox(
        "Model",
        [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
        ],
        index=0,
    )

    st.divider()

    st.markdown(
        """
        **Official source**

        NCCIA Cyber Laws

        The application attempts to download the latest available
        law documents from the official NCCIA cyber-laws page during
        startup.
        """
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚖️ CyberlawGPT</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Pakistan Cyber Law RAG Assistant — ask questions about PECA and
    related official cyber-law documents.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD RAG
# ============================================================

with st.spinner(
    "Downloading official cyber-law documents and building the FAISS index..."
):

    try:

        rag_data = create_rag_index(
            chunk_size=chunk_size,
            overlap=overlap,
        )

        rag_ready = True

    except Exception as error:

        rag_ready = False

        st.error(
            "Cyber-law knowledge base could not be initialized."
        )

        st.code(str(error))

        st.markdown(
            """
            **Possible causes**

            - NCCIA temporarily blocked the request.
            - Government source temporarily unavailable.
            - Internet connection issue.
            - PDF structure changed.
            - Required Python package failed to install.
            """
        )


# ============================================================
# KNOWLEDGE BASE STATUS
# ============================================================

if rag_ready:

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Documents",
            len(rag_data["documents"]),
        )

    with col2:
        st.metric(
            "Legal chunks",
            len(rag_data["chunks"]),
        )

    with col3:
        st.metric(
            "FAISS vectors",
            rag_data["index"].ntotal,
        )

    with col4:
        st.metric(
            "Index",
            rag_data["fingerprint"],
        )

    with st.expander("📚 Loaded legal sources"):

        for document in rag_data["documents"]:

            st.markdown(
                f"""
                **{document['name']}**

                Source: {document['url']}
                """
            )


# ============================================================
# LEGAL DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="warning-box">
    <b>Legal information notice:</b>
    CyberlawGPT provides information based on retrieved Pakistani
    cyber-law documents. It is not a lawyer, does not create an
    attorney-client relationship, and should not replace professional
    legal advice or official legal proceedings.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# ============================================================
# SAMPLE QUESTIONS
# ============================================================

if rag_ready and not st.session_state.messages:

    st.subheader("💡 Try a question")

    examples = [
        "What does PECA say about unauthorized access?",
        "What Pakistani cyber law applies to online fraud?",
        "What does the law say about identity information?",
        "What is cyber stalking under Pakistani law?",
        "What is the role of NCCIA under PECA?",
        "What changed under the 2025 PECA amendment?",
    ]

    cols = st.columns(2)

    for i, example in enumerate(examples):

        with cols[i % 2]:

            if st.button(
                example,
                use_container_width=True,
            ):
                st.session_state.selected_question = example


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander("📚 Retrieved legal sources"):

                for source in message["sources"]:

                    metadata = source["metadata"]

                    st.markdown(
                        f"""
                        **{metadata['document']}**

                        **{metadata['section']}**

                        Similarity: `{source['score']:.3f}`

                        > {source['text'][:1200]}
                        """
                    )


# ============================================================
# QUESTION INPUT
# ============================================================

selected_question = st.session_state.pop(
    "selected_question",
    None,
)

question = st.chat_input(
    "Ask a question about Pakistan's cyber laws..."
)

if selected_question and not question:
    question = selected_question


# ============================================================
# PROCESS QUESTION
# ============================================================

if question and rag_ready:

    question = question.strip()

    if not question:
        st.stop()

    # Display user message.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve relevant law.
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the Pakistani cyber-law knowledge base..."
        ):

            try:

                retrieved = retrieve_documents(
                    question,
                    rag_data,
                    top_k=top_k,
                )

            except Exception as error:

                st.error(
                    f"Retrieval failed: {error}"
                )

                st.stop()

        if not retrieved:

            answer = (
                "I could not find a sufficiently relevant provision "
                "in the loaded cyber-law documents."
            )

            st.markdown(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": [],
                }
            )

        else:

            try:

                with st.spinner(
                    "Analyzing the relevant legal provisions..."
                ):

                    answer = ask_groq(
                        question=question,
                        retrieved=retrieved,
                        technicality=technicality,
                        response_size=response_size,
                        language=language,
                        model_name=groq_model,
                    )

                st.markdown(answer)

                with st.expander(
                    "📚 Retrieved legal sources"
                ):

                    for source in retrieved:

                        metadata = source["metadata"]

                        st.markdown(
                            f"""
                            **{metadata['document']}**

                            **{metadata['section']}**

                            Similarity:
                            `{source['score']:.3f}`

                            > {source['text'][:1200]}
                            """
                        )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved,
                    }
                )

            except Exception as error:

                error_message = (
                    f"Unable to generate the answer: {error}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": retrieved,
                    }
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CyberlawGPT • Python + Streamlit + FAISS + Sentence Transformers + Groq"
)

st.caption(
    "Primary legal source: National Cyber Crime Investigation Agency (NCCIA), Government of Pakistan."
)
