
import os
import requests
import streamlit as st


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. API CONFIGURATION
# ==================================================

def get_api_url():
    """
    Read Render backend URL from Streamlit Secrets.
    Fall back to environment variable or localhost.
    """

    try:
        secret_url = st.secrets.get("API_URL", "")
    except Exception:
        secret_url = ""

    api_url = (
        secret_url
        or os.getenv("API_URL")
        or "http://127.0.0.1:8001"
    )

    return api_url.strip().rstrip("/")


API_URL = get_api_url()


# ==================================================
# 3. SESSION STATE
# ==================================================

defaults = {
    "pdf_uploaded": False,
    "pdf_info": None,
    "chat_history": [],
    "document_name": None,
    "pdf_summary": None,
    "summary_type": "short",
    "selected_page": "Dashboard"
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# 4. CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.stApp {
    background-color: #0B1120;
    color: #F8FAFC;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

.main-heading {
    font-size: 36px;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -1px;
}

.gradient-text {
    color: #60A5FA;
}

.subtitle {
    color: #94A3B8;
    font-size: 15px;
    margin-top: 5px;
}

.section-heading {
    font-size: 21px;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 12px;
}

div[data-testid="stMetric"] {
    background: #172033;
    border: 1px solid #263449;
    padding: 18px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8;
}

div[data-testid="stMetricValue"] {
    color: #F8FAFC;
}

.stButton > button {
    border-radius: 10px;
    min-height: 42px;
    font-weight: 600;
    border: 1px solid #334155;
    transition: 0.2s ease;
}

.stButton > button:hover {
    border-color: #60A5FA;
    color: #60A5FA;
}

.stButton > button[kind="primary"] {
    background: #2563EB;
    border: none;
    color: white;
}

[data-testid="stFileUploader"] {
    background: #172033;
    border: 1px dashed #475569;
    border-radius: 14px;
    padding: 15px;
}

[data-testid="stChatMessage"] {
    background: #172033;
    border: 1px solid #263449;
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 12px;
}

section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263449;
}

hr {
    border-color: #263449;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

div[data-testid="stExpander"] {
    background: #172033;
    border: 1px solid #263449;
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #64748B;
    font-size: 12px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 5. API HELPER FUNCTIONS
# ==================================================

def check_backend():
    """Check whether the Render backend is responding."""

    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=15
        )

        return response.status_code == 200

    except requests.exceptions.RequestException:
        return False


def upload_pdf_to_backend(uploaded_file):
    """Upload PDF and build the FAISS vector database."""

    response = requests.post(
        f"{API_URL}/upload-pdf",
        files={
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf"
            )
        },
        timeout=(15, 600)
    )

    response.raise_for_status()

    return response.json()


def ask_backend(question, chat_history):
    """Ask a question using RAG and conversation history."""

    response = requests.post(
        f"{API_URL}/ask",
        json={
            "question": question,
            "chat_history": chat_history
        },
        timeout=(15, 180)
    )

    response.raise_for_status()

    return response.json()


def summarize_backend(summary_type):
    """Generate a summary of the uploaded PDF."""

    response = requests.post(
        f"{API_URL}/summarize",
        json={
            "summary_type": summary_type
        },
        timeout=(15, 600)
    )

    response.raise_for_status()

    return response.json()


def get_error_message(error):
    """Extract a readable error from API responses."""

    try:
        return error.response.json().get(
            "detail",
            str(error)
        )

    except Exception:
        return str(error)


# ==================================================
# 6. SESSION HELPER FUNCTIONS
# ==================================================

def clear_chat():
    """Clear frontend conversation history."""

    st.session_state.chat_history = []


def reset_document():
    """
    Clear frontend document state.

    Note: This does not clear the backend FAISS index.
    The backend currently has no reset API endpoint.
    """

    st.session_state.pdf_uploaded = False
    st.session_state.pdf_info = None
    st.session_state.chat_history = []
    st.session_state.document_name = None
    st.session_state.pdf_summary = None
    st.session_state.summary_type = "short"


def display_sources(sources):
    """Display source page references."""

    if not sources:
        return

    with st.expander("📖 View PDF Sources"):

        pages = sorted(
            set(
                source.get("page")
                for source in sources
                if source.get("page") is not None
            ),
            key=str
        )

        st.write(
            "**Relevant pages:**",
            ", ".join(map(str, pages))
        )

        for source in sources:

            st.caption(
                f"Page {source.get('page', 'N/A')} | "
                f"Relevance: "
                f"{source.get('relevance_score', 'N/A')}"
            )


# ==================================================
# 7. BACKEND STATUS
# ==================================================

backend_online = check_backend()


# ==================================================
# 8. SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("## 📚 DocuMind AI")

    st.caption("Your intelligent study workspace")

    st.divider()

    st.markdown("### System Status")

    if backend_online:
        st.success("● Backend Connected")
    else:
        st.error("● Backend Offline")

    if st.button(
        "🔄 Refresh Connection",
        use_container_width=True
    ):
        st.rerun()

    st.divider()

    st.markdown("### Navigation")

    st.radio(
        "Go to",
        [
            "Dashboard",
            "AI Chat",
            "PDF Summary"
        ],
        key="selected_page",
        label_visibility="collapsed"
    )

    selected_page = st.session_state.selected_page

    st.divider()

    st.markdown("### Current Document")

    if st.session_state.pdf_uploaded:

        st.success("PDF Ready")

        st.write(
            f"📄 {st.session_state.document_name}"
        )

        if st.session_state.pdf_info:

            st.caption(
                f"{st.session_state.pdf_info.get('total_pages', 0)} pages"
            )

    else:
        st.info("No document uploaded")

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):
        clear_chat()
        st.rerun()

    if st.button(
        "🔄 Clear Local Document View",
        use_container_width=True
    ):
        reset_document()
        st.rerun()

    st.caption(
        "Built with Streamlit • FastAPI • FAISS • OpenRouter"
    )


# ==================================================
# 9. MAIN HEADER
# ==================================================

st.markdown(
    """
    <div class="main-heading">
        Welcome to <span class="gradient-text">DocuMind AI</span>
    </div>

    <div class="subtitle">
        Upload your notes, understand concepts,
        generate summaries and learn smarter.
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")


# ==================================================
# 10. DASHBOARD PAGE
# ==================================================

if selected_page == "Dashboard":

    st.markdown(
        '<div class="section-heading">📂 Document Workspace</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload your study material to get started."
    )

    uploaded_file = st.file_uploader(
        "Upload PDF document",
        type=["pdf"],
        help="Choose your textbook, notes or study material."
    )

    if uploaded_file is not None:

        st.info(
            f"Selected document: {uploaded_file.name}"
        )

        if st.button(
            "🚀 Upload and Process PDF",
            type="primary",
            use_container_width=True
        ):

            if not backend_online:

                st.error(
                    "Backend is offline. Check your Render service."
                )

            else:

                try:

                    with st.spinner(
                        "Processing PDF and generating embeddings..."
                    ):

                        data = upload_pdf_to_backend(
                            uploaded_file
                        )

                    st.session_state.pdf_uploaded = True
                    st.session_state.pdf_info = data
                    st.session_state.document_name = uploaded_file.name

                    st.session_state.chat_history = []
                    st.session_state.pdf_summary = None

                    st.success(
                        "PDF processed successfully!"
                    )

                    st.rerun()

                except requests.exceptions.Timeout:

                    st.error(
                        "Processing timed out. Try a smaller PDF."
                    )

                except requests.exceptions.HTTPError as e:

                    st.error(
                        f"Upload failed: {get_error_message(e)}"
                    )

                except requests.exceptions.RequestException as e:

                    st.error(
                        f"Connection error: {str(e)}"
                    )

                except Exception as e:

                    st.error(
                        f"Unexpected error: {str(e)}"
                    )

    st.divider()

    # DOCUMENT ANALYTICS

    st.markdown(
        '<div class="section-heading">📊 Document Analytics</div>',
        unsafe_allow_html=True
    )

    if st.session_state.pdf_uploaded:

        info = st.session_state.pdf_info or {}

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Pages",
            info.get("total_pages", 0)
        )

        col2.metric(
            "Characters Extracted",
            f"{info.get('total_characters', 0):,}"
        )

        col3.metric(
            "Text Chunks",
            info.get("total_chunks", 0)
        )

        st.success(
            f"Active document: {st.session_state.document_name}"
        )

    else:

        st.info(
            "Your document analytics will appear here "
            "after uploading a PDF."
        )

    st.divider()

    # QUICK ACTIONS

    st.markdown(
        '<div class="section-heading">⚡ Quick Actions</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 💬 AI Chat")

        st.caption(
            "Ask questions directly from your PDF."
        )

        if st.button(
            "Open AI Chat",
            use_container_width=True
        ):

            st.session_state.selected_page = "AI Chat"
            st.rerun()

    with col2:

        st.markdown("### 📝 PDF Summary")

        st.caption(
            "Generate concise or detailed summaries."
        )

        if st.button(
            "Open Summary",
            use_container_width=True
        ):

            st.session_state.selected_page = "PDF Summary"
            st.rerun()

    with col3:

        st.markdown("### 📚 Study Assistant")

        st.caption(
            "Understand difficult concepts using your PDF."
        )

        st.write(
            "Ask follow-up questions in AI Chat."
        )


# ==================================================
# 11. AI CHAT PAGE
# ==================================================

elif selected_page == "AI Chat":

    st.markdown(
        '<div class="section-heading">💬 Chat with Your PDF</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Ask questions and explore your document with AI."
    )

    if not st.session_state.pdf_uploaded:

        st.warning(
            "Please upload and process a PDF from Dashboard first."
        )

    else:

        st.success(
            f"Connected to: {st.session_state.document_name}"
        )

        st.caption(
            f"Conversation messages: "
            f"{len(st.session_state.chat_history)}"
        )

        # CHAT HISTORY

        for message in st.session_state.chat_history:

            with st.chat_message(message["role"]):

                st.markdown(message["content"])

                if message["role"] == "assistant":

                    display_sources(
                        message.get("sources", [])
                    )

        # CHAT INPUT

        question = st.chat_input(
            "Ask anything about your PDF..."
        )

        if question:

            previous_history = [
                {
                    "role": message["role"],
                    "content": message["content"]
                }
                for message in st.session_state.chat_history
                if message["role"] in ("user", "assistant")
            ]

            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching document and generating answer..."
                ):

                    try:

                        result = ask_backend(
                            question,
                            previous_history[-10:]
                        )

                        answer = result.get(
                            "answer",
                            "No answer received."
                        )

                        sources = result.get(
                            "sources",
                            []
                        )

                        st.markdown(answer)

                        display_sources(sources)

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })

                    except requests.exceptions.Timeout:

                        st.error(
                            "Answer generation timed out."
                        )

                    except requests.exceptions.HTTPError as e:

                        st.error(
                            f"API Error: {get_error_message(e)}"
                        )

                    except requests.exceptions.RequestException:

                        st.error(
                            "Cannot connect to backend."
                        )

                    except Exception as e:

                        st.error(
                            f"Unexpected error: {str(e)}"
                        )


# ==================================================
# 12. PDF SUMMARY PAGE
# ==================================================

elif selected_page == "PDF Summary":

    st.markdown(
        '<div class="section-heading">📝 PDF Summarization</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Turn lengthy documents into easy-to-understand study material."
    )

    if not st.session_state.pdf_uploaded:

        st.warning(
            "Upload and process a PDF from Dashboard first."
        )

    else:

        st.success(
            f"Document: {st.session_state.document_name}"
        )

        st.markdown("### Choose Summary Type")

        summary_type = st.selectbox(
            "Summary format",
            options=[
                "short",
                "detailed",
                "key_points"
            ],
            format_func=lambda x: {
                "short": "⚡ Short Summary",
                "detailed": "📖 Detailed Summary",
                "key_points": "📌 Important Key Points"
            }[x],
            key="summary_type"
        )

        descriptions = {
            "short": "A concise overview of the complete document.",
            "detailed": "Detailed explanations of major topics and concepts.",
            "key_points": "Important facts, definitions and concepts in bullet points."
        }

        st.info(
            descriptions[summary_type]
        )

        if st.button(
            "✨ Generate PDF Summary",
            type="primary",
            use_container_width=True
        ):

            if not backend_online:

                st.error("Backend is offline.")

            else:

                try:

                    with st.spinner(
                        "AI is reading your PDF and preparing the summary..."
                    ):

                        result = summarize_backend(
                            summary_type
                        )

                    st.session_state.pdf_summary = result

                    st.success(
                        "Summary generated successfully!"
                    )

                except requests.exceptions.Timeout:

                    st.error(
                        "Summary generation timed out. "
                        "Try again or use a smaller PDF."
                    )

                except requests.exceptions.HTTPError as e:

                    st.error(
                        f"Summary error: {get_error_message(e)}"
                    )

                except requests.exceptions.RequestException:

                    st.error(
                        "Cannot connect to backend."
                    )

                except Exception as e:

                    st.error(
                        f"Unexpected error: {str(e)}"
                    )

        # DISPLAY SUMMARY

        if st.session_state.pdf_summary:

            summary_data = st.session_state.pdf_summary

            st.divider()

            st.markdown("### 📚 Generated Summary")

            col1, col2 = st.columns(2)

            col1.metric(
                "Pages Covered",
                summary_data.get("total_pages", 0)
            )

            col2.metric(
                "Batches Processed",
                summary_data.get("batches_processed", 0)
            )

            summary_text = summary_data.get(
                "summary",
                ""
            )

            st.markdown(summary_text)

            st.download_button(
                label="⬇️ Download Summary",
                data=summary_text,
                file_name="PDF_Summary.txt",
                mime="text/plain",
                use_container_width=True
            )


# ==================================================
# 13. FOOTER
# ==================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        DocuMind AI · Intelligent PDF Study Assistant
        <br>
        Python · Streamlit · FastAPI · FAISS · OpenRouter
    </div>
    """,
    unsafe_allow_html=True
)
