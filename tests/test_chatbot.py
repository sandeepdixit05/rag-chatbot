"""Tests for the main RAG chatbot functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import tempfile

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.chatbot import RAGChatbot, ConversationManager
from src.chatbot.conversation_manager import Message, ConversationSession
from datetime import datetime, timedelta


class TestConversationManager:
    """Tests for ConversationManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ConversationManager(
            max_context_length=1000,
            session_timeout_hours=1,
            persist_conversations=False  # Disable persistence for tests
        )

    def test_create_session(self):
        """Test session creation."""
        session_id = self.manager.create_session()

        assert session_id is not None
        assert session_id in self.manager.sessions
        assert self.manager.get_session(session_id) is not None

    def test_create_session_with_custom_id(self):
        """Test session creation with custom ID."""
        custom_id = "test_session_123"
        session_id = self.manager.create_session(session_id=custom_id)

        assert session_id == custom_id
        assert custom_id in self.manager.sessions

    def test_add_message(self):
        """Test adding messages to session."""
        session_id = self.manager.create_session()

        message_id = self.manager.add_message(
            session_id=session_id,
            role="user",
            content="Hello, world!",
            metadata={"test": True}
        )

        assert message_id is not None

        session = self.manager.get_session(session_id)
        assert len(session.messages) == 1
        assert session.messages[0].content == "Hello, world!"
        assert session.messages[0].role == "user"
        assert session.messages[0].metadata["test"] is True

    def test_add_message_invalid_session(self):
        """Test adding message to invalid session."""
        with pytest.raises(ValueError, match="Session .* not found"):
            self.manager.add_message("invalid_session", "user", "test message")

    def test_conversation_context(self):
        """Test getting conversation context."""
        session_id = self.manager.create_session()

        # Add some messages
        self.manager.add_message(session_id, "user", "First message")
        self.manager.add_message(session_id, "assistant", "First response")
        self.manager.add_message(session_id, "user", "Second message")

        context = self.manager.get_conversation_context(session_id)

        assert len(context) == 3
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "First message"
        assert context[1]["role"] == "assistant"
        assert context[2]["role"] == "user"

    def test_conversation_context_exclude_system(self):
        """Test conversation context excluding system messages."""
        session_id = self.manager.create_session()

        self.manager.add_message(session_id, "system", "System message")
        self.manager.add_message(session_id, "user", "User message")
        self.manager.add_message(session_id, "assistant", "Assistant message")

        context = self.manager.get_conversation_context(
            session_id,
            include_system_messages=False
        )

        assert len(context) == 2
        assert all(msg["role"] != "system" for msg in context)

    def test_conversation_context_max_messages(self):
        """Test conversation context with message limit."""
        session_id = self.manager.create_session()

        # Add many messages
        for i in range(10):
            self.manager.add_message(session_id, "user", f"Message {i}")

        # Test with limit
        context = self.manager.get_conversation_context(session_id)

        # Should be limited by max_conversation_history (20 in our setup)
        assert len(context) <= 20

    def test_session_expiration(self):
        """Test session expiration."""
        session_id = self.manager.create_session()

        # Manually set session as old
        session = self.manager.sessions[session_id]
        session.updated_at = datetime.now() - timedelta(hours=2)

        # Should return None for expired session
        expired_session = self.manager.get_session(session_id)
        assert expired_session is None

    def test_list_active_sessions(self):
        """Test listing active sessions."""
        # Create some sessions
        session1 = self.manager.create_session()
        session2 = self.manager.create_session()
        session3 = self.manager.create_session()

        # Close one session
        self.manager.close_session(session2)

        active = self.manager.list_active_sessions()

        assert len(active) == 2
        assert session1 in active
        assert session3 in active
        assert session2 not in active

    def test_session_summary(self):
        """Test getting session summary."""
        session_id = self.manager.create_session()

        self.manager.add_message(session_id, "user", "User message 1")
        self.manager.add_message(session_id, "assistant", "Assistant response 1")
        self.manager.add_message(session_id, "user", "User message 2")

        summary = self.manager.get_conversation_summary(session_id)

        assert summary["session_id"] == session_id
        assert summary["total_messages"] == 3
        assert summary["user_messages"] == 2
        assert summary["assistant_messages"] == 1
        assert summary["active"] is True

    def test_close_session(self):
        """Test session closure."""
        session_id = self.manager.create_session()

        success = self.manager.close_session(session_id)
        assert success is True

        session = self.manager.get_session(session_id)
        assert session.active is False

    def test_delete_session(self):
        """Test session deletion."""
        session_id = self.manager.create_session()

        success = self.manager.delete_session(session_id)
        assert success is True

        session = self.manager.get_session(session_id)
        assert session is None

    def test_cleanup_expired_sessions(self):
        """Test cleanup of expired sessions."""
        # Create sessions
        session1 = self.manager.create_session()
        session2 = self.manager.create_session()

        # Make one session expired
        session = self.manager.sessions[session1]
        session.updated_at = datetime.now() - timedelta(hours=2)

        # Cleanup expired sessions
        cleaned_count = self.manager.cleanup_expired_sessions()

        assert cleaned_count == 1
        assert self.manager.get_session(session1) is None  # Should be expired
        assert self.manager.get_session(session2) is not None  # Should still exist


class TestRAGChatbot:
    """Tests for RAGChatbot class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'chunk_size': 100,
            'chunk_overlap': 20,
            'max_retrieval_docs': 3,
            'collection_name': 'test_collection',
            'vector_db_path': './test_vector_db',
            'llm_provider': 'openai',
            'llm_model': 'gpt-3.5-turbo',
            'openai_api_key': 'test_key',
            'temperature': 0.7,
            'max_tokens': 1000,
            'embedding_model': 'test-embedding-model'
        }

    @patch('src.chatbot.rag_chatbot.PDFExtractor')
    @patch('src.chatbot.rag_chatbot.DocumentProcessor')
    @patch('src.chatbot.rag_chatbot.EmbeddingGenerator')
    @patch('src.chatbot.rag_chatbot.VectorStore')
    @patch('src.chatbot.rag_chatbot.DocumentRetriever')
    @patch('src.chatbot.rag_chatbot.LLMProvider')
    @patch('src.chatbot.rag_chatbot.ConversationManager')
    def test_chatbot_initialization(
        self,
        mock_conv_mgr,
        mock_llm_provider,
        mock_retriever,
        mock_vector_store,
        mock_embedding_gen,
        mock_doc_processor,
        mock_pdf_extractor
    ):
        """Test chatbot initialization."""

        # Mock embedding generator
        mock_embedding_instance = Mock()
        mock_embedding_instance.embedding_dimension = 384
        mock_embedding_gen.return_value = mock_embedding_instance

        # Mock vector store
        mock_vector_store_instance = Mock()
        mock_vector_store.create_store.return_value = mock_vector_store_instance

        chatbot = RAGChatbot(self.config, initialize_components=True)

        assert chatbot.initialized is True
        assert chatbot.pdf_extractor is not None
        assert chatbot.document_processor is not None
        assert chatbot.embedding_generator is not None
        assert chatbot.vector_store is not None
        assert chatbot.document_retriever is not None
        assert chatbot.llm_provider is not None
        assert chatbot.conversation_manager is not None

    def test_chatbot_uninitialized_load_documents(self):
        """Test load_documents when chatbot is not initialized."""
        chatbot = RAGChatbot(self.config, initialize_components=False)

        with pytest.raises(RuntimeError, match="Chatbot not initialized"):
            chatbot.load_documents("./test_pdfs")

    def test_health_check_uninitialized(self):
        """Test health check on uninitialized chatbot."""
        chatbot = RAGChatbot(self.config, initialize_components=False)

        health = chatbot.health_check()

        assert health["overall"] in ["unhealthy", "degraded"]
        assert health["components"]["initialization"]["status"] == "unhealthy"

    @patch('src.chatbot.rag_chatbot.PDFExtractor')
    @patch('src.chatbot.rag_chatbot.DocumentProcessor')
    @patch('src.chatbot.rag_chatbot.EmbeddingGenerator')
    @patch('src.chatbot.rag_chatbot.VectorStore')
    @patch('src.chatbot.rag_chatbot.DocumentRetriever')
    @patch('src.chatbot.rag_chatbot.LLMProvider')
    @patch('src.chatbot.rag_chatbot.ConversationManager')
    def test_chat_without_documents(
        self,
        mock_conv_mgr,
        mock_llm_provider,
        mock_retriever,
        mock_vector_store,
        mock_embedding_gen,
        mock_doc_processor,
        mock_pdf_extractor
    ):
        """Test chat when no documents are loaded."""

        # Mock components
        mock_embedding_instance = Mock()
        mock_embedding_instance.embedding_dimension = 384
        mock_embedding_gen.return_value = mock_embedding_instance

        mock_vector_store_instance = Mock()
        mock_vector_store.create_store.return_value = mock_vector_store_instance

        chatbot = RAGChatbot(self.config, initialize_components=True)

        # Ensure documents are not loaded
        chatbot.ready_for_chat = False

        response = chatbot.chat("Hello, world!")

        assert "error" in response
        assert "no documents have been loaded" in response["response"].lower()

    @patch('src.chatbot.rag_chatbot.PDFExtractor')
    @patch('src.chatbot.rag_chatbot.DocumentProcessor')
    @patch('src.chatbot.rag_chatbot.EmbeddingGenerator')
    @patch('src.chatbot.rag_chatbot.VectorStore')
    @patch('src.chatbot.rag_chatbot.DocumentRetriever')
    @patch('src.chatbot.rag_chatbot.LLMProvider')
    @patch('src.chatbot.rag_chatbot.ConversationManager')
    def test_successful_chat(
        self,
        mock_conv_mgr,
        mock_llm_provider,
        mock_retriever,
        mock_vector_store,
        mock_embedding_gen,
        mock_doc_processor,
        mock_pdf_extractor
    ):
        """Test successful chat interaction."""

        # Mock components
        mock_embedding_instance = Mock()
        mock_embedding_instance.embedding_dimension = 384
        mock_embedding_gen.return_value = mock_embedding_instance

        mock_vector_store_instance = Mock()
        mock_vector_store.create_store.return_value = mock_vector_store_instance

        # Mock conversation manager
        mock_conv_mgr_instance = Mock()
        mock_conv_mgr_instance.create_session.return_value = "test_session"
        mock_conv_mgr_instance.get_session.return_value = Mock()
        mock_conv_mgr_instance.get_conversation_context.return_value = []
        mock_conv_mgr.return_value = mock_conv_mgr_instance

        # Mock retriever
        mock_retriever_instance = Mock()
        mock_retrieval_result = Mock()
        mock_retrieval_result.content = "Test content"
        mock_retrieval_result.source_info = {"file_name": "test.pdf", "page_number": 1}
        mock_retrieval_result.chunk_id = "chunk_1"
        mock_retrieval_result.score = 0.8
        mock_retriever_instance.retrieve.return_value = [mock_retrieval_result]
        mock_retriever.return_value = mock_retriever_instance

        # Mock LLM provider
        mock_llm_instance = Mock()
        mock_llm_response = Mock()
        mock_llm_response.content = "This is a test response"
        mock_llm_response.model = "gpt-3.5-turbo"
        mock_llm_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_llm_response.response_time = 1.5
        mock_llm_instance.generate_response.return_value = mock_llm_response
        mock_llm_provider.create_provider.return_value = mock_llm_instance

        chatbot = RAGChatbot(self.config, initialize_components=True)
        chatbot.ready_for_chat = True  # Set as ready

        response = chatbot.chat("What is the test content about?")

        assert "error" not in response
        assert response["response"] == "This is a test response"
        assert response["session_id"] == "test_session"
        assert len(response["sources"]) == 1
        assert response["model"] == "gpt-3.5-turbo"

    def test_format_source(self):
        """Test source formatting."""
        chatbot = RAGChatbot(self.config, initialize_components=False)

        # Mock retrieval result
        mock_result = Mock()
        mock_result.source_info = {
            "file_name": "test_document.pdf",
            "page_number": 42
        }
        mock_result.chunk_id = "chunk_123"
        mock_result.score = 0.85
        mock_result.content = "This is test content for the source formatting test."

        formatted = chatbot._format_source(mock_result)

        assert formatted["document"] == "test_document.pdf"
        assert formatted["page"] == 42
        assert formatted["chunk_id"] == "chunk_123"
        assert formatted["score"] == 0.85
        assert "test content" in formatted["content_preview"]

    def test_get_system_stats_uninitialized(self):
        """Test system stats when uninitialized."""
        chatbot = RAGChatbot(self.config, initialize_components=False)

        stats = chatbot.get_system_stats()

        assert "error" in stats
        assert stats["error"] == "System not initialized"


class TestMessageAndSession:
    """Tests for Message and ConversationSession classes."""

    def test_message_creation(self):
        """Test message creation and serialization."""
        timestamp = datetime.now()
        message = Message(
            role="user",
            content="Test message",
            timestamp=timestamp,
            message_id="msg_123",
            metadata={"key": "value"}
        )

        assert message.role == "user"
        assert message.content == "Test message"
        assert message.timestamp == timestamp
        assert message.message_id == "msg_123"
        assert message.metadata["key"] == "value"

    def test_message_to_dict(self):
        """Test message dictionary conversion."""
        timestamp = datetime.now()
        message = Message(
            role="assistant",
            content="Assistant response",
            timestamp=timestamp,
            message_id="msg_456"
        )

        msg_dict = message.to_dict()

        assert msg_dict["role"] == "assistant"
        assert msg_dict["content"] == "Assistant response"
        assert msg_dict["timestamp"] == timestamp.isoformat()
        assert msg_dict["message_id"] == "msg_456"
        assert msg_dict["metadata"] == {}

    def test_message_from_dict(self):
        """Test message creation from dictionary."""
        timestamp = datetime.now()
        msg_dict = {
            "role": "user",
            "content": "Test content",
            "timestamp": timestamp.isoformat(),
            "message_id": "msg_789",
            "metadata": {"test": True}
        }

        message = Message.from_dict(msg_dict)

        assert message.role == "user"
        assert message.content == "Test content"
        assert message.timestamp == timestamp
        assert message.message_id == "msg_789"
        assert message.metadata["test"] is True

    def test_conversation_session_creation(self):
        """Test conversation session creation."""
        session_id = "session_123"
        created_at = datetime.now()
        messages = []
        metadata = {"user_id": "user_456"}

        session = ConversationSession(
            session_id=session_id,
            created_at=created_at,
            updated_at=created_at,
            messages=messages,
            metadata=metadata
        )

        assert session.session_id == session_id
        assert session.created_at == created_at
        assert session.messages == messages
        assert session.metadata == metadata
        assert session.active is True

    def test_conversation_session_add_message(self):
        """Test adding message to conversation session."""
        session = ConversationSession(
            session_id="test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            messages=[],
            metadata={}
        )

        original_update_time = session.updated_at

        # Add a small delay to ensure time difference
        import time
        time.sleep(0.01)

        message = Message(
            role="user",
            content="Test message",
            timestamp=datetime.now(),
            message_id="msg_1"
        )

        session.add_message(message)

        assert len(session.messages) == 1
        assert session.messages[0] == message
        assert session.updated_at > original_update_time

    def test_conversation_session_get_recent_messages(self):
        """Test getting recent messages from session."""
        messages = [
            Message("user", f"Message {i}", datetime.now(), f"msg_{i}")
            for i in range(10)
        ]

        session = ConversationSession(
            session_id="test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            messages=messages,
            metadata={}
        )

        # Test limit
        recent = session.get_recent_messages(5)
        assert len(recent) == 5
        assert recent[0].content == "Message 5"  # Should be the last 5 messages

        # Test no limit
        all_recent = session.get_recent_messages(0)
        assert len(all_recent) == 10


if __name__ == "__main__":
    pytest.main([__file__])