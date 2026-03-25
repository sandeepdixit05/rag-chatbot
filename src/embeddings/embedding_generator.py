"""Generate embeddings for text chunks using various embedding models."""

import logging
from typing import Dict, List, Optional, Union, Any
import numpy as np
from pathlib import Path
import json
import time

try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    SentenceTransformer = None
    torch = None

try:
    import openai
except ImportError:
    openai = None

from .document_processor import DocumentChunk

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using various providers and models."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        provider: str = "sentence_transformers",
        api_key: Optional[str] = None,
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        """Initialize embedding generator.

        Args:
            model_name: Name of the embedding model
            provider: Provider type ("sentence_transformers", "openai", "huggingface")
            api_key: API key for cloud providers
            batch_size: Batch size for processing
            device: Device to use for local models ("cpu", "cuda")
        """
        self.model_name = model_name
        self.provider = provider.lower()
        self.api_key = api_key
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")

        self.model = None
        self.embedding_dimension = None

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the embedding model based on provider."""
        logger.info(f"Initializing {self.provider} model: {self.model_name}")

        if self.provider == "sentence_transformers":
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")

            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                # Get embedding dimension
                test_embedding = self.model.encode(["test"], show_progress_bar=False)
                self.embedding_dimension = len(test_embedding[0])
                logger.info(f"Loaded sentence-transformers model with dimension: {self.embedding_dimension}")

            except Exception as e:
                logger.error(f"Failed to load sentence-transformers model: {str(e)}")
                raise

        elif self.provider == "openai":
            if openai is None:
                raise ImportError("openai not installed. Install with: pip install openai")

            if not self.api_key:
                raise ValueError("OpenAI API key required for OpenAI embeddings")

            # Initialize OpenAI client
            openai.api_key = self.api_key
            self.model = openai  # Store client reference

            # Set embedding dimension based on model
            if "text-embedding-3-large" in self.model_name:
                self.embedding_dimension = 3072
            elif "text-embedding-3-small" in self.model_name:
                self.embedding_dimension = 1536
            elif "text-embedding-ada-002" in self.model_name:
                self.embedding_dimension = 1536
            else:
                self.embedding_dimension = 1536  # Default

            logger.info(f"Initialized OpenAI embeddings with dimension: {self.embedding_dimension}")

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True,
        normalize: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        logger.info(f"Generating embeddings for {len(texts)} texts")

        if self.provider == "sentence_transformers":
            return self._generate_sentence_transformers_embeddings(
                texts, show_progress, normalize
            )
        elif self.provider == "openai":
            return self._generate_openai_embeddings(texts, normalize)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_sentence_transformers_embeddings(
        self,
        texts: List[str],
        show_progress: bool,
        normalize: bool
    ) -> List[List[float]]:
        """Generate embeddings using sentence-transformers."""
        try:
            # Process in batches to manage memory
            all_embeddings = []

            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]

                batch_embeddings = self.model.encode(
                    batch_texts,
                    show_progress_bar=show_progress and i == 0,  # Show progress only for first batch
                    normalize_embeddings=normalize,
                    convert_to_tensor=False
                )

                all_embeddings.extend(batch_embeddings.tolist())

                if show_progress:
                    logger.debug(f"Processed batch {i // self.batch_size + 1}/{(len(texts) - 1) // self.batch_size + 1}")

            logger.info(f"Generated {len(all_embeddings)} embeddings")
            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating sentence-transformers embeddings: {str(e)}")
            raise

    def _generate_openai_embeddings(
        self,
        texts: List[str],
        normalize: bool
    ) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            all_embeddings = []

            # Process in smaller batches for API rate limits
            api_batch_size = min(self.batch_size, 100)  # OpenAI limit

            for i in range(0, len(texts), api_batch_size):
                batch_texts = texts[i:i + api_batch_size]

                # Add retry logic for API calls
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = openai.embeddings.create(
                            model=self.model_name,
                            input=batch_texts
                        )

                        batch_embeddings = [item.embedding for item in response.data]

                        if normalize:
                            # Normalize embeddings
                            batch_embeddings = [
                                self._normalize_embedding(emb) for emb in batch_embeddings
                            ]

                        all_embeddings.extend(batch_embeddings)
                        break

                    except Exception as api_error:
                        if attempt == max_retries - 1:
                            raise api_error
                        logger.warning(f"API retry {attempt + 1}/{max_retries}: {str(api_error)}")
                        time.sleep(2 ** attempt)  # Exponential backoff

                logger.debug(f"Processed OpenAI batch {i // api_batch_size + 1}/{(len(texts) - 1) // api_batch_size + 1}")

            logger.info(f"Generated {len(all_embeddings)} OpenAI embeddings")
            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {str(e)}")
            raise

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize an embedding vector."""
        embedding_array = np.array(embedding)
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            return (embedding_array / norm).tolist()
        return embedding

    def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate embeddings for document chunks.

        Args:
            chunks: List of document chunks
            show_progress: Whether to show progress

        Returns:
            List of dictionaries containing chunk data and embeddings
        """
        if not chunks:
            return []

        logger.info(f"Embedding {len(chunks)} document chunks")

        # Extract texts from chunks
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings
        embeddings = self.generate_embeddings(texts, show_progress)

        # Combine chunks with embeddings
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunks.append({
                'content': chunk.content,
                'metadata': chunk.metadata,
                'chunk_id': chunk.chunk_id,
                'embedding': embedding,
                'embedding_model': self.model_name,
                'embedding_dimension': self.embedding_dimension
            })

        logger.info(f"Successfully embedded {len(embedded_chunks)} chunks")
        return embedded_chunks

    def save_embeddings(
        self,
        embedded_chunks: List[Dict[str, Any]],
        output_path: Union[str, Path]
    ) -> None:
        """Save embeddings to file.

        Args:
            embedded_chunks: List of embedded chunks
            output_path: Path to save embeddings
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving embeddings to {output_path}")

        # Convert numpy arrays to lists for JSON serialization
        serializable_chunks = []
        for chunk in embedded_chunks:
            serializable_chunk = chunk.copy()
            if isinstance(serializable_chunk['embedding'], np.ndarray):
                serializable_chunk['embedding'] = serializable_chunk['embedding'].tolist()
            serializable_chunks.append(serializable_chunk)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'embeddings': serializable_chunks,
                'model_info': {
                    'model_name': self.model_name,
                    'provider': self.provider,
                    'embedding_dimension': self.embedding_dimension
                },
                'total_chunks': len(embedded_chunks)
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(embedded_chunks)} embeddings")

    def load_embeddings(self, input_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load embeddings from file.

        Args:
            input_path: Path to load embeddings from

        Returns:
            List of embedded chunks
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {input_path}")

        logger.info(f"Loading embeddings from {input_path}")

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        embedded_chunks = data.get('embeddings', [])
        model_info = data.get('model_info', {})

        logger.info(f"Loaded {len(embedded_chunks)} embeddings (model: {model_info.get('model_name', 'unknown')})")

        return embedded_chunks

    def compute_similarity(
        self,
        query_embedding: List[float],
        doc_embeddings: List[List[float]],
        method: str = "cosine"
    ) -> List[float]:
        """Compute similarity between query and document embeddings.

        Args:
            query_embedding: Query embedding vector
            doc_embeddings: List of document embedding vectors
            method: Similarity method ("cosine", "dot_product")

        Returns:
            List of similarity scores
        """
        query_array = np.array(query_embedding)
        doc_arrays = np.array(doc_embeddings)

        if method == "cosine":
            # Cosine similarity
            dot_products = np.dot(doc_arrays, query_array)
            query_norm = np.linalg.norm(query_array)
            doc_norms = np.linalg.norm(doc_arrays, axis=1)

            similarities = dot_products / (doc_norms * query_norm)
        elif method == "dot_product":
            # Dot product similarity
            similarities = np.dot(doc_arrays, query_array)
        else:
            raise ValueError(f"Unsupported similarity method: {method}")

        return similarities.tolist()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'provider': self.provider,
            'embedding_dimension': self.embedding_dimension,
            'device': self.device,
            'batch_size': self.batch_size
        }