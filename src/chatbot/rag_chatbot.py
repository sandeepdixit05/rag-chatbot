"""Main RAG chatbot implementation combining all components."""

import logging
from typing import Dict, List, Optional, Any, Union, Iterator
from pathlib import Path
import time

from ..pdf_processor import PDFExtractor
from ..embeddings import DocumentProcessor, EmbeddingGenerator
from ..retrieval import VectorStore, DocumentRetriever
from ..llm import LLMProvider, PromptTemplates
from .conversation_manager import ConversationManager

logger = logging.getLogger(__name__)


class RAGChatbot:
    """Main RAG chatbot that combines all components."""

    def __init__(
        self,
        config: Dict[str, Any],
        initialize_components: bool = True
    ):
        """Initialize the RAG chatbot.

        Args:
            config: Configuration dictionary
            initialize_components: Whether to initialize all components immediately
        """
        self.config = config
        self.initialized = False

        # Component instances
        self.pdf_extractor = None
        self.document_processor = None
        self.embedding_generator = None
        self.vector_store = None
        self.document_retriever = None
        self.llm_provider = None
        self.conversation_manager = None
        self.prompt_templates = None

        # State tracking
        self.documents_loaded = False
        self.ready_for_chat = False

        if initialize_components:
            self.initialize()

    def initialize(self) -> None:
        """Initialize all chatbot components."""
        logger.info("Initializing RAG chatbot components...")

        try:
            # Initialize prompt templates
            self.prompt_templates = PromptTemplates()

            # Initialize PDF extractor
            self.pdf_extractor = PDFExtractor(use_cache=True)

            # Initialize document processor
            self.document_processor = DocumentProcessor(
                chunk_size=self.config.get('chunk_size', 1000),
                chunk_overlap=self.config.get('chunk_overlap', 200),
                splitter_type="recursive"
            )

            # Initialize embedding generator
            self.embedding_generator = EmbeddingGenerator(
                model_name=self.config.get('embedding_model', 'sentence-transformers/all-mpnet-base-v2'),
                provider="sentence_transformers"
            )

            # Initialize vector store
            self.vector_store = VectorStore.create_store(
                store_type="chroma",
                collection_name=self.config.get('collection_name', 'pdf_documents'),
                persist_directory=self.config.get('vector_db_path', './data/vector_db'),
                embedding_dimension=self.embedding_generator.embedding_dimension
            )

            # Initialize document retriever
            self.document_retriever = DocumentRetriever(
                vector_store=self.vector_store,
                embedding_generator=self.embedding_generator,
                default_top_k=self.config.get('max_retrieval_docs', 5)
            )

            # Initialize LLM provider
            self.llm_provider = LLMProvider.create_provider(
                provider_type=self.config.get('llm_provider', 'openai'),
                api_key=self.config.get('openai_api_key') or self.config.get('anthropic_api_key'),
                model=self.config.get('llm_model', 'gpt-4o-mini')
            )

            # Initialize conversation manager
            self.conversation_manager = ConversationManager(
                max_context_length=self.config.get('max_tokens', 2000),
                persist_conversations=True
            )

            self.initialized = True
            logger.info("RAG chatbot initialization completed successfully")

            # Check if documents are already loaded
            stats = self.vector_store.get_collection_stats()
            if stats.get('total_documents', 0) > 0:
                self.documents_loaded = True
                self.ready_for_chat = True
                logger.info(f"Found {stats['total_documents']} documents already loaded in vector store")

        except Exception as e:
            logger.error(f"Error initializing RAG chatbot: {str(e)}")
            raise

    def load_documents(self, pdf_directory: Union[str, Path]) -> Dict[str, Any]:
        """Load and process PDF documents.

        Args:
            pdf_directory: Path to directory containing PDF files

        Returns:
            Processing statistics
        """
        if not self.initialized:
            raise RuntimeError("Chatbot not initialized. Call initialize() first.")

        logger.info(f"Loading documents from: {pdf_directory}")
        start_time = time.time()

        try:
            # Extract text from PDFs
            pdf_results = self.pdf_extractor.extract_from_directory(
                pdf_directory,
                method="auto",
                clean_text=True
            )

            if not pdf_results:
                return {
                    "success": False,
                    "error": "No PDF files found or processed",
                    "processing_time": time.time() - start_time
                }

            # Process documents into chunks
            all_chunks = []
            for pdf_result in pdf_results:
                chunks = self.document_processor.process_pdf_extraction(
                    pdf_result,
                    preserve_page_info=True
                )
                all_chunks.extend(chunks)

            # Filter chunks
            filtered_chunks = self.document_processor.filter_chunks(
                all_chunks,
                min_length=50,
                remove_short_sentences=True
            )

            logger.info(f"Created {len(filtered_chunks)} chunks from {len(pdf_results)} PDFs")

            # Generate embeddings
            embedded_chunks = self.embedding_generator.embed_chunks(
                filtered_chunks,
                show_progress=True
            )

            # Add to vector store
            document_ids = self.vector_store.add_documents(embedded_chunks)

            processing_time = time.time() - start_time

            # Update state
            self.documents_loaded = True
            self.ready_for_chat = True

            # Get processing statistics
            stats = {
                "success": True,
                "pdfs_processed": len(pdf_results),
                "total_chunks": len(all_chunks),
                "filtered_chunks": len(filtered_chunks),
                "embedded_chunks": len(embedded_chunks),
                "documents_added": len(document_ids),
                "processing_time": processing_time,
                "chunk_statistics": self.document_processor.get_chunk_statistics(filtered_chunks)
            }

            logger.info(f"Document loading completed in {processing_time:.2f} seconds")
            return stats

        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        stream_response: bool = False,
        include_sources: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a chat message and generate response.

        Args:
            message: User message
            session_id: Conversation session ID
            stream_response: Whether to stream the response
            include_sources: Whether to include source citations
            **kwargs: Additional parameters

        Returns:
            Response dictionary
        """
        if not self.ready_for_chat:
            return {
                "response": "I'm sorry, but no documents have been loaded yet. Please load some PDF documents first.",
                "session_id": session_id,
                "sources": [],
                "error": "No documents loaded"
            }

        try:
            # Create or get session
            if not session_id:
                session_id = self.conversation_manager.create_session()
            elif not self.conversation_manager.get_session(session_id):
                session_id = self.conversation_manager.create_session(session_id)

            # Add user message to conversation
            self.conversation_manager.add_message(session_id, "user", message)

            # Retrieve relevant documents
            retrieval_results = self.document_retriever.retrieve(
                query=message,
                top_k=kwargs.get('max_docs', self.config.get('max_retrieval_docs', 5)),
                rerank_query=kwargs.get('rerank', True)
            )

            # Get conversation context
            conversation_context = self.conversation_manager.get_conversation_context(
                session_id,
                include_system_messages=False,
                max_tokens=kwargs.get('max_context_tokens', 1000)
            )

            # Prepare context and prompt
            if retrieval_results:
                # Format context from retrieval results
                context_string = self.prompt_templates.create_context_string(
                    retrieval_results,
                    include_sources=include_sources,
                    max_context_length=kwargs.get('max_context_length', 3000)
                )

                # Choose appropriate template based on conversation history
                if len(conversation_context) > 2:  # Has conversation history
                    conversation_history = self.prompt_templates.create_conversation_history(
                        conversation_context[:-1]  # Exclude current message
                    )
                    prompt = self.prompt_templates.format_template(
                        "conversation_rag",
                        conversation_history=conversation_history,
                        context=context_string,
                        question=message
                    )
                else:
                    # Use basic RAG template
                    template_name = "rag_answer_with_sources" if include_sources else "rag_answer"
                    prompt = self.prompt_templates.format_template(
                        template_name,
                        context=context_string,
                        question=message
                    )

            else:
                # No relevant context found
                prompt = self.prompt_templates.format_template(
                    "no_context_response",
                    question=message
                )

            # Generate system message
            system_message = self.prompt_templates.format_template(
                "system_message",
                session_context=f"Session {session_id} with {len(conversation_context)} messages"
            )

            # Generate response
            if stream_response:
                streaming_response = self.llm_provider.generate_streaming_response(
                    prompt=prompt,
                    system_message=system_message,
                    temperature=self.config.get('temperature', 0.7),
                    max_tokens=self.config.get('max_tokens', 2000)
                )

                # Handle streaming response
                response_content = ""
                for chunk in streaming_response.content_stream:
                    response_content += chunk
                    # In a real streaming implementation, you'd yield these chunks

                # Create response object for streaming
                response = {
                    "response": response_content,
                    "session_id": session_id,
                    "sources": [self._format_source(result) for result in retrieval_results] if include_sources else [],
                    "streaming": True,
                    "model": streaming_response.model,
                    "retrieval_stats": {
                        "documents_found": len(retrieval_results),
                        "search_query": message
                    }
                }

            else:
                llm_response = self.llm_provider.generate_response(
                    prompt=prompt,
                    system_message=system_message,
                    temperature=self.config.get('temperature', 0.7),
                    max_tokens=self.config.get('max_tokens', 2000)
                )

                response = {
                    "response": llm_response.content,
                    "session_id": session_id,
                    "sources": [self._format_source(result) for result in retrieval_results] if include_sources else [],
                    "model": llm_response.model,
                    "usage": llm_response.usage,
                    "response_time": llm_response.response_time,
                    "retrieval_stats": {
                        "documents_found": len(retrieval_results),
                        "search_query": message
                    }
                }

            # Add assistant response to conversation
            self.conversation_manager.add_message(
                session_id,
                "assistant",
                response["response"],
                metadata={
                    "sources": response["sources"],
                    "model": response.get("model"),
                    "retrieval_stats": response["retrieval_stats"]
                }
            )

            return response

        except Exception as e:
            logger.error(f"Error processing chat message: {str(e)}")

            error_response = self.prompt_templates.format_template(
                "error_handling",
                error_type="Processing Error",
                question=message,
                error_description=f"processing your request: {str(e)}",
                additional_help="Please try again or contact support if the issue persists."
            )

            return {
                "response": error_response,
                "session_id": session_id,
                "sources": [],
                "error": str(e)
            }

    def _format_source(self, retrieval_result) -> Dict[str, Any]:
        """Format a retrieval result as a source citation."""
        source_info = getattr(retrieval_result, 'source_info', {}) or {}

        return {
            "document": source_info.get('file_name', 'Unknown'),
            "page": source_info.get('page_number'),
            "chunk_id": getattr(retrieval_result, 'chunk_id', ''),
            "score": getattr(retrieval_result, 'score', 0.0),
            "content_preview": getattr(retrieval_result, 'content', '')[:200] + "..." if getattr(retrieval_result, 'content', '') else ""
        }

    def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """Get conversation history for a session."""
        session = self.conversation_manager.get_session(session_id)
        if not session:
            return {"error": "Session not found"}

        return {
            "session_id": session_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                }
                for msg in session.messages
            ],
            "summary": self.conversation_manager.get_conversation_summary(session_id)
        }

    def list_active_sessions(self) -> List[str]:
        """List all active conversation sessions."""
        if not self.initialized:
            return []

        return self.conversation_manager.list_active_sessions()

    def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session."""
        if not self.initialized:
            return False

        return self.conversation_manager.delete_session(session_id)

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        if not self.initialized:
            return {"error": "System not initialized"}

        stats = {
            "initialized": self.initialized,
            "documents_loaded": self.documents_loaded,
            "ready_for_chat": self.ready_for_chat,
            "vector_store": self.vector_store.get_collection_stats() if self.vector_store else {},
            "embedding_model": self.embedding_generator.get_model_info() if self.embedding_generator else {},
            "llm_model": self.llm_provider.get_model_info() if self.llm_provider else {},
            "conversation_manager": self.conversation_manager.get_stats() if self.conversation_manager else {}
        }

        return stats

    def reset_system(self, keep_documents: bool = True) -> Dict[str, Any]:
        """Reset the chatbot system.

        Args:
            keep_documents: Whether to keep loaded documents

        Returns:
            Reset operation result
        """
        try:
            # Clear conversations
            if self.conversation_manager:
                active_sessions = self.conversation_manager.list_active_sessions()
                for session_id in active_sessions:
                    self.conversation_manager.delete_session(session_id)

            # Clear vector store if requested
            if not keep_documents and self.vector_store:
                if hasattr(self.vector_store, 'reset_collection'):
                    self.vector_store.reset_collection()
                    self.documents_loaded = False
                    self.ready_for_chat = False

            logger.info(f"System reset completed (keep_documents: {keep_documents})")

            return {
                "success": True,
                "message": "System reset successfully",
                "documents_kept": keep_documents,
                "ready_for_chat": self.ready_for_chat
            }

        except Exception as e:
            logger.error(f"Error during system reset: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of all components."""
        health_status = {
            "overall": "healthy",
            "components": {},
            "timestamp": time.time()
        }

        try:
            # Check initialization
            health_status["components"]["initialization"] = {
                "status": "healthy" if self.initialized else "unhealthy",
                "ready_for_chat": self.ready_for_chat
            }

            # Check vector store
            if self.vector_store:
                try:
                    stats = self.vector_store.get_collection_stats()
                    health_status["components"]["vector_store"] = {
                        "status": "healthy",
                        "document_count": stats.get("total_documents", 0)
                    }
                except Exception as e:
                    health_status["components"]["vector_store"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }

            # Check LLM provider
            if self.llm_provider:
                try:
                    model_info = self.llm_provider.get_model_info()
                    health_status["components"]["llm_provider"] = {
                        "status": "healthy",
                        "provider": model_info.get("provider"),
                        "model": model_info.get("model")
                    }
                except Exception as e:
                    health_status["components"]["llm_provider"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }

            # Check conversation manager
            if self.conversation_manager:
                try:
                    stats = self.conversation_manager.get_stats()
                    health_status["components"]["conversation_manager"] = {
                        "status": "healthy",
                        "active_sessions": stats.get("active_sessions", 0)
                    }
                except Exception as e:
                    health_status["components"]["conversation_manager"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }

            # Determine overall health
            unhealthy_components = [
                comp for comp, status in health_status["components"].items()
                if status.get("status") != "healthy"
            ]

            if unhealthy_components:
                health_status["overall"] = "degraded" if len(unhealthy_components) < len(health_status["components"]) else "unhealthy"
                health_status["unhealthy_components"] = unhealthy_components

        except Exception as e:
            health_status["overall"] = "unhealthy"
            health_status["error"] = str(e)

        return health_status