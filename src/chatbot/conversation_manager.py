"""Conversation memory and context management for the RAG chatbot."""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a conversation message."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    message_id: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message_id=data["message_id"],
            metadata=data.get("metadata", {})
        )


@dataclass
class ConversationSession:
    """Represents a conversation session."""

    session_id: str
    created_at: datetime
    updated_at: datetime
    messages: List[Message]
    metadata: Dict[str, Any]
    active: bool = True

    def add_message(self, message: Message) -> None:
        """Add a message to the session."""
        self.messages.append(message)
        self.updated_at = datetime.now()

    def get_recent_messages(self, limit: int = 10) -> List[Message]:
        """Get recent messages from the session."""
        return self.messages[-limit:] if limit > 0 else self.messages

    def get_message_count(self) -> int:
        """Get total message count."""
        return len(self.messages)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata,
            "active": self.active
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSession":
        """Create session from dictionary."""
        return cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            messages=[Message.from_dict(msg) for msg in data["messages"]],
            metadata=data["metadata"],
            active=data.get("active", True)
        )


class ConversationManager:
    """Manages conversation sessions and context."""

    def __init__(
        self,
        max_context_length: int = 4000,
        max_conversation_history: int = 20,
        session_timeout_hours: int = 24,
        persist_conversations: bool = True,
        storage_path: str = "./data/conversations"
    ):
        """Initialize conversation manager.

        Args:
            max_context_length: Maximum context length for LLM
            max_conversation_history: Maximum number of messages to keep in context
            session_timeout_hours: Hours after which inactive sessions expire
            persist_conversations: Whether to persist conversations to disk
            storage_path: Path to store conversation data
        """
        self.max_context_length = max_context_length
        self.max_conversation_history = max_conversation_history
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self.persist_conversations = persist_conversations
        self.storage_path = Path(storage_path)

        # In-memory session storage
        self.sessions: Dict[str, ConversationSession] = {}

        # Create storage directory
        if self.persist_conversations:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load_sessions()

        logger.info(f"Initialized conversation manager with {len(self.sessions)} sessions")

    def create_session(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new conversation session.

        Args:
            session_id: Optional session ID (generated if not provided)
            metadata: Session metadata

        Returns:
            Session ID
        """
        session_id = session_id or str(uuid.uuid4())
        metadata = metadata or {}

        if session_id in self.sessions:
            logger.warning(f"Session {session_id} already exists")
            return session_id

        session = ConversationSession(
            session_id=session_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            messages=[],
            metadata=metadata
        )

        self.sessions[session_id] = session

        if self.persist_conversations:
            self._save_session(session)

        logger.info(f"Created new session: {session_id}")
        return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a message to a session.

        Args:
            session_id: Session ID
            role: Message role ("user", "assistant", "system")
            content: Message content
            metadata: Message metadata

        Returns:
            Message ID
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        message_id = str(uuid.uuid4())
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            message_id=message_id,
            metadata=metadata
        )

        session.add_message(message)

        if self.persist_conversations:
            self._save_session(session)

        logger.debug(f"Added message to session {session_id}: {role}")
        return message_id

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get a conversation session.

        Args:
            session_id: Session ID

        Returns:
            Conversation session or None
        """
        session = self.sessions.get(session_id)

        # Check if session is expired
        if session and self._is_session_expired(session):
            self._expire_session(session_id)
            return None

        return session

    def get_conversation_context(
        self,
        session_id: str,
        include_system_messages: bool = True,
        max_tokens: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Get conversation context for LLM.

        Args:
            session_id: Session ID
            include_system_messages: Whether to include system messages
            max_tokens: Maximum tokens to include (rough estimate)

        Returns:
            List of message dictionaries for LLM context
        """
        session = self.get_session(session_id)
        if not session:
            return []

        messages = session.get_recent_messages(self.max_conversation_history)

        # Filter messages if needed
        if not include_system_messages:
            messages = [msg for msg in messages if msg.role != "system"]

        # Convert to LLM format
        context = []
        total_length = 0

        for msg in reversed(messages):  # Start from most recent
            msg_dict = {"role": msg.role, "content": msg.content}
            msg_length = len(msg.content)

            # Check token limit (rough estimate: 4 chars = 1 token)
            if max_tokens and (total_length + msg_length) > (max_tokens * 4):
                break

            context.insert(0, msg_dict)  # Insert at beginning to maintain order
            total_length += msg_length

        logger.debug(f"Retrieved {len(context)} messages for context (session: {session_id})")
        return context

    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the conversation.

        Args:
            session_id: Session ID

        Returns:
            Conversation summary
        """
        session = self.get_session(session_id)
        if not session:
            return {}

        user_messages = [msg for msg in session.messages if msg.role == "user"]
        assistant_messages = [msg for msg in session.messages if msg.role == "assistant"]

        return {
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "total_messages": len(session.messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "duration_minutes": (session.updated_at - session.created_at).total_seconds() / 60,
            "active": session.active,
            "metadata": session.metadata
        }

    def list_active_sessions(self) -> List[str]:
        """List all active session IDs."""
        active_sessions = []

        for session_id, session in self.sessions.items():
            if not self._is_session_expired(session) and session.active:
                active_sessions.append(session_id)

        return active_sessions

    def close_session(self, session_id: str) -> bool:
        """Close a conversation session.

        Args:
            session_id: Session ID

        Returns:
            True if session was closed
        """
        session = self.get_session(session_id)
        if not session:
            return False

        session.active = False
        session.updated_at = datetime.now()

        if self.persist_conversations:
            self._save_session(session)

        logger.info(f"Closed session: {session_id}")
        return True

    def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session.

        Args:
            session_id: Session ID

        Returns:
            True if session was deleted
        """
        if session_id not in self.sessions:
            return False

        del self.sessions[session_id]

        # Delete from storage
        if self.persist_conversations:
            session_file = self.storage_path / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()

        logger.info(f"Deleted session: {session_id}")
        return True

    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions.

        Returns:
            Number of sessions removed
        """
        expired_sessions = []

        for session_id, session in self.sessions.items():
            if self._is_session_expired(session):
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            self._expire_session(session_id)

        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        return len(expired_sessions)

    def export_session(self, session_id: str, output_path: str) -> bool:
        """Export session to file.

        Args:
            session_id: Session ID
            output_path: Output file path

        Returns:
            True if export was successful
        """
        session = self.get_session(session_id)
        if not session:
            return False

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

            logger.info(f"Exported session {session_id} to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting session: {str(e)}")
            return False

    def import_session(self, input_path: str) -> Optional[str]:
        """Import session from file.

        Args:
            input_path: Input file path

        Returns:
            Session ID if import was successful
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            session = ConversationSession.from_dict(data)
            self.sessions[session.session_id] = session

            if self.persist_conversations:
                self._save_session(session)

            logger.info(f"Imported session {session.session_id} from {input_path}")
            return session.session_id

        except Exception as e:
            logger.error(f"Error importing session: {str(e)}")
            return None

    def _is_session_expired(self, session: ConversationSession) -> bool:
        """Check if a session is expired."""
        return datetime.now() - session.updated_at > self.session_timeout

    def _expire_session(self, session_id: str) -> None:
        """Expire a session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.active = False

            if self.persist_conversations:
                self._save_session(session)
            else:
                del self.sessions[session_id]

            logger.debug(f"Expired session: {session_id}")

    def _save_session(self, session: ConversationSession) -> None:
        """Save session to storage."""
        try:
            session_file = self.storage_path / f"{session.session_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Error saving session {session.session_id}: {str(e)}")

    def _load_sessions(self) -> None:
        """Load sessions from storage."""
        if not self.storage_path.exists():
            return

        for session_file in self.storage_path.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                session = ConversationSession.from_dict(data)

                # Only load active, non-expired sessions
                if session.active and not self._is_session_expired(session):
                    self.sessions[session.session_id] = session

            except Exception as e:
                logger.warning(f"Error loading session from {session_file}: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get conversation manager statistics."""
        active_sessions = len(self.list_active_sessions())
        total_messages = sum(len(session.messages) for session in self.sessions.values())

        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "total_messages": total_messages,
            "average_messages_per_session": total_messages / max(1, len(self.sessions)),
            "max_context_length": self.max_context_length,
            "max_conversation_history": self.max_conversation_history,
            "session_timeout_hours": self.session_timeout.total_seconds() / 3600,
            "persist_conversations": self.persist_conversations
        }