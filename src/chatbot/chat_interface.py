"""Chat interface implementations for different platforms."""

import logging
from typing import Dict, List, Optional, Any, Callable
import asyncio
from pathlib import Path
import json

try:
    import streamlit as st
except ImportError:
    st = None

try:
    import gradio as gr
except ImportError:
    gr = None

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError:
    FastAPI = None
    HTTPException = None
    BaseModel = None
    StreamingResponse = None

from .rag_chatbot import RAGChatbot

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """Chat message model for API."""
    message: str
    session_id: Optional[str] = None
    include_sources: bool = True
    stream: bool = False


class ChatResponse(BaseModel):
    """Chat response model for API."""
    response: str
    session_id: str
    sources: List[Dict[str, Any]]
    model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    response_time: Optional[float] = None


class ChatInterface:
    """Base class for chat interfaces."""

    def __init__(self, chatbot: RAGChatbot):
        """Initialize chat interface.

        Args:
            chatbot: RAG chatbot instance
        """
        self.chatbot = chatbot

    def process_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a chat message."""
        return self.chatbot.chat(message, session_id, **kwargs)


class StreamlitInterface(ChatInterface):
    """Streamlit-based chat interface."""

    def __init__(self, chatbot: RAGChatbot, title: str = "RAG Chatbot"):
        """Initialize Streamlit interface.

        Args:
            chatbot: RAG chatbot instance
            title: Application title
        """
        if st is None:
            raise ImportError("streamlit not installed. Install with: pip install streamlit")

        super().__init__(chatbot)
        self.title = title

    def run(self):
        """Run the Streamlit interface."""
        st.set_page_config(
            page_title=self.title,
            page_icon="🤖",
            layout="wide"
        )

        st.title(self.title)

        # Sidebar for system information and controls
        self._render_sidebar()

        # Main chat interface
        self._render_chat_interface()

    def _render_sidebar(self):
        """Render the sidebar with system controls."""
        st.sidebar.title("System Status")

        # Health check
        health = self.chatbot.health_check()
        status_color = "🟢" if health["overall"] == "healthy" else "🟡" if health["overall"] == "degraded" else "🔴"
        st.sidebar.markdown(f"**Status:** {status_color} {health['overall'].title()}")

        # System stats
        stats = self.chatbot.get_system_stats()
        if stats.get("vector_store", {}).get("total_documents"):
            st.sidebar.metric("Documents Loaded", stats["vector_store"]["total_documents"])

        # Document upload
        st.sidebar.subheader("📄 Load Documents")
        uploaded_files = st.sidebar.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            key="pdf_uploader"
        )

        if uploaded_files and st.sidebar.button("Process Documents"):
            self._process_uploaded_files(uploaded_files)

        # Session management
        st.sidebar.subheader("💬 Sessions")
        active_sessions = self.chatbot.list_active_sessions()
        if active_sessions:
            selected_session = st.sidebar.selectbox(
                "Select Session",
                options=["New Session"] + active_sessions,
                key="session_selector"
            )
            if selected_session != "New Session":
                st.session_state.current_session_id = selected_session
        else:
            st.sidebar.info("No active sessions")

        # Settings
        st.sidebar.subheader("⚙️ Settings")
        st.sidebar.slider("Max Documents", 1, 10, 5, key="max_docs")
        st.sidebar.checkbox("Include Sources", True, key="include_sources")
        st.sidebar.checkbox("Stream Response", False, key="stream_response")

    def _render_chat_interface(self):
        """Render the main chat interface."""
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_session_id" not in st.session_state:
            st.session_state.current_session_id = None

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show sources if available
                if message.get("sources") and st.session_state.get("include_sources", True):
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**{i}. {source.get('document', 'Unknown')}**")
                            if source.get('page'):
                                st.markdown(f"Page: {source['page']}")
                            if source.get('content_preview'):
                                st.markdown(f"_{source['content_preview']}_")

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self.process_message(
                        message=prompt,
                        session_id=st.session_state.current_session_id,
                        stream_response=st.session_state.get("stream_response", False),
                        include_sources=st.session_state.get("include_sources", True),
                        max_docs=st.session_state.get("max_docs", 5)
                    )

                    if response.get("error"):
                        st.error(f"Error: {response['error']}")
                        return

                    # Update session ID
                    st.session_state.current_session_id = response.get("session_id")

                    # Display response
                    st.markdown(response["response"])

                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["response"],
                        "sources": response.get("sources", [])
                    })

                    # Show sources
                    if response.get("sources") and st.session_state.get("include_sources", True):
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(response["sources"], 1):
                                st.markdown(f"**{i}. {source.get('document', 'Unknown')}**")
                                if source.get('page'):
                                    st.markdown(f"Page: {source['page']}")
                                if source.get('content_preview'):
                                    st.markdown(f"_{source['content_preview']}_")

    def _process_uploaded_files(self, uploaded_files):
        """Process uploaded PDF files."""
        if not uploaded_files:
            return

        # Create temporary directory for uploaded files
        temp_dir = Path("./data/temp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save uploaded files
            saved_files = []
            for uploaded_file in uploaded_files:
                file_path = temp_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                saved_files.append(file_path)

            # Process documents
            with st.spinner(f"Processing {len(saved_files)} documents..."):
                result = self.chatbot.load_documents(temp_dir)

            if result.get("success"):
                st.sidebar.success(f"✅ Processed {result['pdfs_processed']} PDFs")
                st.sidebar.info(f"Created {result['embedded_chunks']} chunks")
            else:
                st.sidebar.error(f"❌ Error: {result.get('error', 'Unknown error')}")

            # Clean up temporary files
            for file_path in saved_files:
                file_path.unlink(missing_ok=True)

        except Exception as e:
            st.sidebar.error(f"Error processing files: {str(e)}")


class GradioInterface(ChatInterface):
    """Gradio-based chat interface."""

    def __init__(self, chatbot: RAGChatbot, title: str = "RAG Chatbot"):
        """Initialize Gradio interface.

        Args:
            chatbot: RAG chatbot instance
            title: Application title
        """
        if gr is None:
            raise ImportError("gradio not installed. Install with: pip install gradio")

        super().__init__(chatbot)
        self.title = title

    def create_interface(self) -> gr.Interface:
        """Create Gradio interface."""

        def chat_fn(message, history, session_id, include_sources, max_docs):
            """Process chat message for Gradio."""
            if not message.strip():
                return history, "", session_id

            # Add user message to history
            history.append([message, ""])

            # Generate response
            response = self.process_message(
                message=message,
                session_id=session_id or None,
                include_sources=include_sources,
                max_docs=max_docs
            )

            if response.get("error"):
                assistant_response = f"Error: {response['error']}"
            else:
                assistant_response = response["response"]

                # Add sources if available
                if response.get("sources") and include_sources:
                    sources_text = "\n\n**Sources:**\n"
                    for i, source in enumerate(response["sources"], 1):
                        sources_text += f"{i}. {source.get('document', 'Unknown')}"
                        if source.get('page'):
                            sources_text += f" (Page {source['page']})"
                        sources_text += "\n"
                    assistant_response += sources_text

            # Update history
            history[-1][1] = assistant_response

            return history, "", response.get("session_id", session_id)

        def upload_files(files):
            """Handle file uploads."""
            if not files:
                return "No files uploaded"

            try:
                # Create temporary directory
                temp_dir = Path("./data/temp_uploads")
                temp_dir.mkdir(parents=True, exist_ok=True)

                # Save files
                for file in files:
                    file_path = temp_dir / Path(file.name).name
                    with open(file_path, "wb") as f:
                        f.write(file.read())

                # Process documents
                result = self.chatbot.load_documents(temp_dir)

                if result.get("success"):
                    return f"✅ Successfully processed {result['pdfs_processed']} PDFs, created {result['embedded_chunks']} chunks"
                else:
                    return f"❌ Error: {result.get('error', 'Unknown error')}"

            except Exception as e:
                return f"Error processing files: {str(e)}"

        # Create interface
        with gr.Blocks(title=self.title) as interface:
            gr.Markdown(f"# {self.title}")

            with gr.Row():
                with gr.Column(scale=3):
                    # Chat interface
                    chatbot_component = gr.Chatbot(label="Conversation")
                    message_input = gr.Textbox(
                        label="Message",
                        placeholder="Ask a question about your documents...",
                        lines=2
                    )

                    with gr.Row():
                        submit_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear")

                with gr.Column(scale=1):
                    # Controls
                    gr.Markdown("### Settings")
                    session_id_input = gr.Textbox(
                        label="Session ID",
                        placeholder="Leave empty for new session"
                    )
                    include_sources = gr.Checkbox(label="Include Sources", value=True)
                    max_docs = gr.Slider(
                        label="Max Documents",
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1
                    )

                    # File upload
                    gr.Markdown("### Upload Documents")
                    file_upload = gr.File(
                        label="PDF Files",
                        file_types=[".pdf"],
                        file_count="multiple"
                    )
                    upload_btn = gr.Button("Process Documents")
                    upload_status = gr.Textbox(label="Upload Status", interactive=False)

                    # System status
                    gr.Markdown("### System Status")
                    status_display = gr.JSON(label="Health Check", value=self.chatbot.health_check())

            # Event handlers
            submit_btn.click(
                fn=chat_fn,
                inputs=[message_input, chatbot_component, session_id_input, include_sources, max_docs],
                outputs=[chatbot_component, message_input, session_id_input]
            )

            message_input.submit(
                fn=chat_fn,
                inputs=[message_input, chatbot_component, session_id_input, include_sources, max_docs],
                outputs=[chatbot_component, message_input, session_id_input]
            )

            clear_btn.click(
                fn=lambda: ([], "", ""),
                outputs=[chatbot_component, message_input, session_id_input]
            )

            upload_btn.click(
                fn=upload_files,
                inputs=[file_upload],
                outputs=[upload_status]
            )

        return interface

    def run(self, **kwargs):
        """Run the Gradio interface."""
        interface = self.create_interface()
        interface.launch(**kwargs)


class FastAPIInterface(ChatInterface):
    """FastAPI-based REST API interface."""

    def __init__(self, chatbot: RAGChatbot, title: str = "RAG Chatbot API"):
        """Initialize FastAPI interface.

        Args:
            chatbot: RAG chatbot instance
            title: API title
        """
        if FastAPI is None:
            raise ImportError("fastapi not installed. Install with: pip install fastapi uvicorn")

        super().__init__(chatbot)
        self.title = title
        self.app = FastAPI(title=title, description="RAG Chatbot REST API")

        self._setup_routes()

    def _setup_routes(self):
        """Set up API routes."""

        @self.app.post("/chat", response_model=ChatResponse)
        async def chat_endpoint(message: ChatMessage):
            """Chat endpoint."""
            try:
                response = self.process_message(
                    message=message.message,
                    session_id=message.session_id,
                    include_sources=message.include_sources,
                    stream_response=message.stream
                )

                if response.get("error"):
                    raise HTTPException(status_code=500, detail=response["error"])

                return ChatResponse(
                    response=response["response"],
                    session_id=response["session_id"],
                    sources=response.get("sources", []),
                    model=response.get("model"),
                    usage=response.get("usage"),
                    response_time=response.get("response_time")
                )

            except Exception as e:
                logger.error(f"Chat endpoint error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health")
        async def health_endpoint():
            """Health check endpoint."""
            return self.chatbot.health_check()

        @self.app.get("/stats")
        async def stats_endpoint():
            """System statistics endpoint."""
            return self.chatbot.get_system_stats()

        @self.app.get("/sessions")
        async def list_sessions_endpoint():
            """List active sessions."""
            return {"sessions": self.chatbot.list_active_sessions()}

        @self.app.get("/sessions/{session_id}")
        async def get_session_endpoint(session_id: str):
            """Get session history."""
            history = self.chatbot.get_session_history(session_id)
            if "error" in history:
                raise HTTPException(status_code=404, detail=history["error"])
            return history

        @self.app.delete("/sessions/{session_id}")
        async def delete_session_endpoint(session_id: str):
            """Delete a session."""
            success = self.chatbot.delete_session(session_id)
            if not success:
                raise HTTPException(status_code=404, detail="Session not found")
            return {"message": "Session deleted successfully"}

        @self.app.post("/upload")
        async def upload_documents_endpoint(files: List[UploadFile] = File(...)):
            """Upload and process PDF documents."""
            try:
                # Create temporary directory
                temp_dir = Path("./data/temp_uploads")
                temp_dir.mkdir(parents=True, exist_ok=True)

                # Save uploaded files
                saved_files = []
                for file in files:
                    if not file.filename.lower().endswith('.pdf'):
                        continue

                    file_path = temp_dir / file.filename
                    with open(file_path, "wb") as f:
                        content = await file.read()
                        f.write(content)
                    saved_files.append(file_path)

                if not saved_files:
                    raise HTTPException(status_code=400, detail="No valid PDF files uploaded")

                # Process documents
                result = self.chatbot.load_documents(temp_dir)

                # Clean up
                for file_path in saved_files:
                    file_path.unlink(missing_ok=True)

                return result

            except Exception as e:
                logger.error(f"Upload endpoint error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    def run(self, host: str = "127.0.0.1", port: int = 8000, **kwargs):
        """Run the FastAPI server."""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port, **kwargs)