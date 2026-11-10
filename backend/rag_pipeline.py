
# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================

import os

# Use PyTorch for Sentence Transformers
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"


# ==========================================
# 2. IMPORTS
# ==========================================

import faiss
import numpy as np

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from backend.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LLM_MODEL,
    EMBEDDING_MODEL,
    VECTOR_DB_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


# ==========================================
# 3. INITIALIZE MODELS AND VARIABLES
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL,
    device="cpu"
)

print("Embedding model loaded successfully!")


if OPENROUTER_API_KEY:

    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL
    )

else:
    client = None


# Active document data

vector_index = None
document_chunks = []
document_pages = []


# ==========================================
# 4. TEXT CHUNKING
# ==========================================

def create_chunks(pages):
    """
    Divide extracted PDF text into smaller chunks.
    Preserve original page numbers.
    """

    chunks = []

    if CHUNK_SIZE <= 0:
        raise ValueError(
            "CHUNK_SIZE must be greater than zero."
        )

    if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be non-negative "
            "and smaller than CHUNK_SIZE."
        )

    step = CHUNK_SIZE - CHUNK_OVERLAP

    for page in pages:

        text = page.get("content", "")
        page_number = page.get("page")

        if not isinstance(text, str) or not text.strip():
            continue

        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            start += step

    return chunks


# ==========================================
# 5. GENERATE EMBEDDINGS
# ==========================================

def generate_embeddings(chunks):
    """
    Convert text chunks into numerical vectors
    using Sentence Transformers.
    """

    if not chunks:
        raise ValueError(
            "No text chunks available for embeddings."
        )

    texts = [
        chunk["text"] for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Invalid values found in generated embeddings."
        )

    return embeddings


# ==========================================
# 6. BUILD FAISS VECTOR DATABASE
# ==========================================

def build_vector_store(pages):
    """
    Create embeddings and store them in FAISS.
    Preserve complete PDF text for summarization.
    """

    global vector_index
    global document_chunks
    global document_pages

    if not pages:
        raise ValueError(
            "No readable text found in PDF."
        )

    # Step 1: Create chunks
    new_chunks = create_chunks(pages)

    if not new_chunks:
        raise ValueError(
            "Could not create text chunks. "
            "Check whether your PDF contains selectable text."
        )

    # Step 2: Generate embeddings
    embeddings = generate_embeddings(new_chunks)

    # Step 3: Create FAISS index
    dimension = embeddings.shape[1]

    new_index = faiss.IndexFlatIP(dimension)

    new_index.add(embeddings)

    # Step 4: Save FAISS index
    index_directory = os.path.dirname(VECTOR_DB_PATH)

    if index_directory:
        os.makedirs(index_directory, exist_ok=True)

    faiss.write_index(
        new_index,
        VECTOR_DB_PATH
    )

    # Step 5: Update active document
    vector_index = new_index

    document_chunks = new_chunks

    document_pages = [
        page.copy() for page in pages
    ]

    print(
        f"FAISS index created with "
        f"{len(document_chunks)} chunks."
    )

    return {
        "total_chunks": len(document_chunks),
        "message": (
            "FAISS vector database created successfully."
        )
    }


# ==========================================
# 7. RETRIEVE RELEVANT CHUNKS
# ==========================================

def retrieve_context(question, top_k=4):
    """
    Find the most relevant PDF chunks
    using FAISS similarity search.
    """

    if vector_index is None or not document_chunks:

        raise ValueError(
            "Please upload and process a PDF first."
        )

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    if top_k <= 0:

        raise ValueError(
            "top_k must be greater than zero."
        )

    # Generate question embedding
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    # Search FAISS
    number_of_results = min(
        top_k,
        len(document_chunks)
    )

    scores, indices = vector_index.search(
        question_embedding,
        number_of_results
    )

    results = []

    for score, index in zip(scores[0], indices[0]):

        if index == -1:
            continue

        chunk = document_chunks[index]

        results.append({
            "text": chunk["text"],
            "page": chunk["page"],
            "score": float(score)
        })

    return results


# ==========================================
# 8. GENERATE ANSWER USING OPENROUTER
# ==========================================

def generate_answer(question, chat_history=None):
    """
    Retrieve relevant PDF context and generate
    an answer using OpenRouter.

    Supports conversational chat history.
    """

    if client is None:

        raise ValueError(
            "OpenRouter API key is missing. "
            "Check your .env file."
        )

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    question = question.strip()

    # ======================================
    # STEP 1: VALIDATE CHAT HISTORY
    # ======================================

    valid_history = []

    for message in (chat_history or [])[-10:]:

        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content")

        if (
            role in ("user", "assistant")
            and isinstance(content, str)
            and content.strip()
        ):

            valid_history.append({
                "role": role,
                "content": content.strip()[:2000]
            })

    # ======================================
    # STEP 2: BUILD RETRIEVAL QUERY
    # ======================================

    # Recent user questions help FAISS understand
    # follow-up questions such as:
    # "Explain it again"
    # "Give me an example"
    # "What are its advantages?"

    previous_user_questions = [
        message["content"]
        for message in valid_history
        if message["role"] == "user"
    ]

    recent_questions = previous_user_questions[-2:]

    retrieval_query = " ".join(
        recent_questions + [question]
    )

    # ======================================
    # STEP 3: RETRIEVE PDF CONTEXT
    # ======================================

    retrieved_chunks = retrieve_context(
        retrieval_query,
        top_k=4
    )

    if not retrieved_chunks:

        return {
            "answer": (
                "I could not find relevant information "
                "in the uploaded PDF."
            ),
            "sources": []
        }

    # ======================================
    # STEP 4: PREPARE PDF CONTEXT
    # ======================================

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"[Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    # ======================================
    # STEP 5: PREPARE CURRENT PROMPT
    # ======================================

    prompt = f"""
You are an intelligent PDF study assistant.

Your task is to answer the user's LATEST question
using the provided PDF context.

RULES:

1. Give clear and easy-to-understand answers.
2. Use the PDF as the source of factual information.
3. Do not invent facts.
4. If information is unavailable in the PDF,
   clearly mention that.
5. Mention relevant PDF page numbers.
6. Use examples only when supported by the PDF.
7. Use previous conversation to understand
   follow-up questions.
8. Answer the latest question, not an older question.
9. Avoid unnecessary repetition.

PDF CONTEXT:

{context}

LATEST USER QUESTION:

{question}
"""

    # ======================================
    # STEP 6: PREPARE CHAT MESSAGES
    # ======================================

    messages = [

        {
            "role": "system",
            "content": (
                "You are a helpful and accurate "
                "PDF question-answering assistant. "
                "Use PDF context for factual answers. "
                "Use chat history only to understand "
                "conversation and follow-up questions."
            )
        }

    ]

    # Add previous conversation
    messages.extend(valid_history)

    # Add latest question with retrieved PDF context
    messages.append({
        "role": "user",
        "content": prompt
    })

    # ======================================
    # STEP 7: CALL OPENROUTER API
    # ======================================

    try:

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2
        )

        answer = response.choices[0].message.content

        if not answer:

            raise ValueError(
                "OpenRouter returned an empty answer."
            )

    except Exception as e:

        raise RuntimeError(
            f"OpenRouter API request failed: {str(e)}"
        ) from e

    # ======================================
    # STEP 8: PREPARE SOURCE CITATIONS
    # ======================================

    sources = [
        {
            "page": chunk["page"],
            "relevance_score": round(
                chunk["score"], 4
            )
        }
        for chunk in retrieved_chunks
    ]

    return {
        "answer": answer,
        "sources": sources
    }


# ==========================================
# 9. PDF SUMMARIZATION
# ==========================================

def summarize_document(summary_type="short"):
    """
    Generate a summary of the complete uploaded PDF.

    Supported types:
    1. short
    2. detailed
    3. key_points
    """

    if client is None:

        raise ValueError(
            "OpenRouter API key is missing. "
            "Check your .env file."
        )

    if not document_pages:

        raise ValueError(
            "Please upload a PDF before generating a summary."
        )

    valid_types = [
        "short",
        "detailed",
        "key_points"
    ]

    if summary_type not in valid_types:

        raise ValueError(
            f"Invalid summary type. Choose from {valid_types}."
        )

    # ======================================
    # STEP 1: SUMMARY INSTRUCTIONS
    # ======================================

    summary_instructions = {

        "short": """
Create a concise summary of the PDF.

Cover its main topic and most important ideas.
Use simple language.
Avoid unnecessary details.
""",

        "detailed": """
Create a detailed and well-organized summary.

Explain the major topics, concepts and subtopics.
Preserve important definitions, explanations,
examples and technical terms from the PDF.

Use clear headings and subheadings.
""",

        "key_points": """
Extract the most important points from the PDF.

Present them as bullet points.

Include important definitions, formulas,
facts and concepts wherever present.
"""
    }

    # ======================================
    # STEP 2: PREPARE PDF TEXT
    # ======================================

    full_text = []

    for page in document_pages:

        page_number = page.get("page", "Unknown")

        page_content = page.get("content", "")

        if (
            isinstance(page_content, str)
            and page_content.strip()
        ):

            full_text.append(
                f"[Page {page_number}]\n{page_content}"
            )

    if not full_text:

        raise ValueError(
            "No readable text found in the uploaded PDF."
        )

    # ======================================
    # STEP 3: CREATE SUMMARY BATCHES
    # ======================================

    MAX_BATCH_CHARS = 12000

    batches = []

    current_batch = ""

    for page_text in full_text:

        remaining_text = page_text

        while remaining_text:

            available_space = (
                MAX_BATCH_CHARS - len(current_batch)
            )

            if available_space <= 0:

                batches.append(current_batch)

                current_batch = ""

                continue

            part = remaining_text[:available_space]

            current_batch += part + "\n\n"

            remaining_text = remaining_text[
                available_space:
            ]

            if len(current_batch) >= MAX_BATCH_CHARS:

                batches.append(current_batch)

                current_batch = ""

    if current_batch.strip():

        batches.append(current_batch)

    print(
        f"Generating summary using {len(batches)} batches..."
    )

    # ======================================
    # STEP 4: SUMMARIZE EACH BATCH
    # ======================================

    partial_summaries = []

    for batch_number, batch_text in enumerate(
        batches,
        start=1
    ):

        prompt = f"""
You are an expert PDF summarization assistant.

Summarize the following PDF content.

SUMMARY TYPE:

{summary_type}

INSTRUCTIONS:

{summary_instructions[summary_type]}

IMPORTANT RULES:

1. Use only the provided PDF content.
2. Do not invent information.
3. Preserve important technical terms.
4. Do not omit important concepts.
5. Keep the summary well structured.
6. Preserve page references where useful.

PDF CONTENT:

{batch_text}
"""

        try:

            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an accurate document "
                            "summarization assistant."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2
            )

            batch_summary = (
                response.choices[0].message.content
            )

            if not batch_summary:

                raise ValueError(
                    "Empty summary received from OpenRouter."
                )

            partial_summaries.append(batch_summary)

            print(
                f"Batch {batch_number}/"
                f"{len(batches)} completed."
            )

        except Exception as e:

            raise RuntimeError(
                f"Summary generation failed at batch "
                f"{batch_number}: {str(e)}"
            ) from e

    # ======================================
    # STEP 5: COMBINE PARTIAL SUMMARIES
    # ======================================

    if len(partial_summaries) == 1:

        final_summary = partial_summaries[0]

    else:

        combined_text = "\n\n".join(
            partial_summaries
        )

        final_prompt = f"""
You are an expert document summarization assistant.

The following are partial summaries of different
sections of the SAME PDF document.

Combine them into one complete and coherent summary.

SUMMARY TYPE:

{summary_type}

INSTRUCTIONS:

{summary_instructions[summary_type]}

RULES:

1. Remove unnecessary repetition.
2. Preserve all important information.
3. Maintain logical topic order wherever possible.
4. Do not introduce information not present
   in the partial summaries.
5. Do not omit important concepts.
6. Use clear headings and formatting.

PARTIAL SUMMARIES:

{combined_text}
"""

        try:

            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You combine partial document "
                            "summaries accurately."
                        )
                    },
                    {
                        "role": "user",
                        "content": final_prompt
                    }
                ],
                temperature=0.2
            )

            final_summary = (
                response.choices[0].message.content
            )

            if not final_summary:

                raise ValueError(
                    "Final summary is empty."
                )

        except Exception as e:

            raise RuntimeError(
                f"Final summary generation failed: {str(e)}"
            ) from e

    # ======================================
    # STEP 6: RETURN SUMMARY
    # ======================================

    return {
        "summary": final_summary,
        "summary_type": summary_type,
        "total_pages": len(document_pages),
        "batches_processed": len(batches)
    }


# ==========================================
# 10. RESET VECTOR DATABASE
# ==========================================

def reset_vector_store():
    """
    Clear the currently active document,
    FAISS index, chunks and summary data.
    """

    global vector_index
    global document_chunks
    global document_pages

    vector_index = None

    document_chunks = []

    document_pages = []

    if os.path.exists(VECTOR_DB_PATH):

        os.remove(VECTOR_DB_PATH)

    return {
        "message": "Vector database reset successfully."
    }
