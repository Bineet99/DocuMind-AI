
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.pdf_service import (
    save_pdf,
    extract_text,
    get_pdf_info
)

from backend.rag_pipeline import (
    build_vector_store,
    generate_answer,
    summarize_document
)


# ==========================================
# 1. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="RAG PDF Assistant API",
    description=(
        "PDF Question Answering using RAG, "
        "FAISS, OpenRouter, Chat Memory "
        "and PDF Summarization"
    ),
    version="1.2.0"
)


# ==========================================
# 2. CORS CONFIGURATION
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 3. REQUEST MODELS
# ==========================================

class ChatMessage(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=5000
    )

    chat_history: list[ChatMessage] = Field(
        default_factory=list
    )


class SummaryRequest(BaseModel):
    summary_type: str = Field(
        default="short",
        pattern="^(short|detailed|key_points)$"
    )


# ==========================================
# 4. HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "RAG PDF Assistant API is running!",
        "version": "1.2.0",
        "features": [
            "PDF Upload",
            "Text Extraction",
            "FAISS Vector Search",
            "OpenRouter Answer Generation",
            "Conversational Chat Memory",
            "PDF Source Citations",
            "PDF Summarization",
            "Short Summary",
            "Detailed Summary",
            "Key Points Extraction"
        ]
    }


# ==========================================
# 5. PDF UPLOAD ENDPOINT
# ==========================================

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...)
):
    """
    Upload PDF, extract text, create embeddings,
    and build the FAISS vector database.
    """

    if (
        not file.filename
        or not file.filename.lower().endswith(".pdf")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    try:

        # Step 1: Save uploaded PDF
        file_path = save_pdf(file)

        # Step 2: Extract text with page numbers
        pages = extract_text(file_path)

        if not pages:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No readable text found. "
                    "This PDF may require OCR."
                )
            )

        # Step 3: Get PDF metadata
        pdf_info = get_pdf_info(file_path)

        # Step 4: Build FAISS vector database
        rag_info = build_vector_store(pages)

        # Step 5: Count extracted characters
        total_characters = sum(
            len(page.get("content", ""))
            for page in pages
        )

        # Step 6: Return processing details
        return {
            "message": (
                "PDF uploaded and processed successfully!"
            ),
            "filename": pdf_info["filename"],
            "total_pages": pdf_info["total_pages"],
            "total_characters": total_characters,
            "total_chunks": rag_info["total_chunks"],
            "status": "ready"
        }

    except HTTPException:
        raise

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(e)}"
        )

    finally:
        await file.close()


# ==========================================
# 6. ASK QUESTION WITH CHAT MEMORY
# ==========================================

@app.post("/ask")
def ask_question(request: AskRequest):
    """
    Answer questions using PDF context and
    recent conversation history.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    chat_history = []

    for message in request.chat_history:

        if (
            message.role in ("user", "assistant")
            and message.content.strip()
        ):
            chat_history.append({
                "role": message.role,
                "content": message.content.strip()
            })

    # Keep last 10 messages
    chat_history = chat_history[-10:]

    try:

        result = generate_answer(
            question=question,
            chat_history=chat_history
        )

        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "memory_used": len(chat_history) > 0,
            "history_messages": len(chat_history)
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation failed: {str(e)}"
        )


# ==========================================
# 7. PDF SUMMARIZATION ENDPOINT
# ==========================================

@app.post("/summarize")
def summarize_pdf(request: SummaryRequest):
    """
    Generate a summary of the uploaded PDF.

    Available summary types:
    - short
    - detailed
    - key_points
    """

    try:

        result = summarize_document(
            summary_type=request.summary_type
        )

        return {
            "message": "PDF summary generated successfully!",
            "summary": result["summary"],
            "summary_type": result["summary_type"],
            "total_pages": result["total_pages"],
            "batches_processed": result["batches_processed"]
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF summarization failed: {str(e)}"
        )


# ==========================================
# 8. HEALTH CHECK ENDPOINT
# ==========================================

@app.get("/health")
def health_check():
    """
    Check whether the FastAPI application
    is responding.
    """

    return {
        "status": "healthy",
        "service": "RAG PDF Assistant",
        "version": "1.2.0"
    }

