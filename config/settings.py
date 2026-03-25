"""Configuration settings for the RAG chatbot."""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Model Settings
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    llm_provider: str = "openai"  # "openai", "anthropic", or "local"
    llm_model: str = "gpt-4o-mini"  # Default OpenAI model

    # Vector Database Settings
    vector_db_path: str = "./data/vector_db"
    collection_name: str = "pdf_documents"

    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retrieval_docs: int = 5

    # PDF Processing
    pdf_input_dir: str = "./data/pdfs"
    processed_dir: str = "./data/processed"

    # Application Settings
    app_name: str = "RAG Chatbot"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000

    # Temperature for LLM responses
    temperature: float = 0.7
    max_tokens: int = 2000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()