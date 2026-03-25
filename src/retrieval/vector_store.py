"""Vector database implementation using Chroma and FAISS."""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import json
import numpy as np
from abc import ABC, abstractmethod

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

try:
    import faiss
except ImportError:
    faiss = None

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_documents(self, embedded_chunks: List[Dict[str, Any]]) -> List[str]:
        """Add embedded documents to the vector store."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector store."""
        pass

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        pass


class ChromaVectorStore(BaseVectorStore):
    """Vector store implementation using ChromaDB."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./data/vector_db",
        embedding_dimension: Optional[int] = None
    ):
        """Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            embedding_dimension: Dimension of embeddings (for validation)
        """
        if chromadb is None:
            raise ImportError("chromadb not installed. Install with: pip install chromadb")

        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.embedding_dimension = embedding_dimension

        # Create persist directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except ValueError:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"Created new collection: {collection_name}")

    def add_documents(self, embedded_chunks: List[Dict[str, Any]]) -> List[str]:
        """Add embedded documents to ChromaDB.

        Args:
            embedded_chunks: List of chunks with embeddings and metadata

        Returns:
            List of document IDs added
        """
        if not embedded_chunks:
            return []

        logger.info(f"Adding {len(embedded_chunks)} documents to ChromaDB")

        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in embedded_chunks:
            ids.append(chunk['chunk_id'])
            embeddings.append(chunk['embedding'])
            documents.append(chunk['content'])

            # Prepare metadata (ChromaDB requires string values)
            metadata = chunk['metadata'].copy()
            for key, value in metadata.items():
                if not isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)

            metadatas.append(metadata)

        try:
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Successfully added {len(ids)} documents to ChromaDB")
            return ids

        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {str(e)}")
            raise

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents in ChromaDB.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Metadata filters

        Returns:
            List of similar documents with scores
        """
        try:
            # Prepare where clause for filtering
            where_clause = filter_dict if filter_dict else None

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )

            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i],
                        'chunk_id': results['ids'][0][i],
                        'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'distance': results['distances'][0][i]
                    })

            logger.debug(f"Found {len(formatted_results)} similar documents")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching ChromaDB: {str(e)}")
            raise

    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from ChromaDB.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            True if successful
        """
        try:
            self.collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents from ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {str(e)}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection."""
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'total_documents': count,
                'persist_directory': str(self.persist_directory)
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}

    def reset_collection(self):
        """Reset the collection (delete all documents)."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Reset collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            raise


class FAISSVectorStore(BaseVectorStore):
    """Vector store implementation using FAISS."""

    def __init__(
        self,
        embedding_dimension: int,
        index_type: str = "IVF",
        persist_directory: str = "./data/vector_db",
        collection_name: str = "documents"
    ):
        """Initialize FAISS vector store.

        Args:
            embedding_dimension: Dimension of embeddings
            index_type: Type of FAISS index ("Flat", "IVF", "HNSW")
            persist_directory: Directory to persist the index
            collection_name: Name of the collection
        """
        if faiss is None:
            raise ImportError("faiss not installed. Install with: pip install faiss-cpu")

        self.embedding_dimension = embedding_dimension
        self.index_type = index_type
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name

        # Create persist directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize FAISS index
        self._create_index()

        # Store document metadata separately
        self.documents = {}  # id -> document data
        self.id_to_index = {}  # document_id -> faiss index position
        self.index_to_id = {}  # faiss index position -> document_id
        self.next_index = 0

        # Load existing data if available
        self._load_index()

    def _create_index(self):
        """Create FAISS index based on type."""
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product (cosine for normalized vectors)
        elif self.index_type == "IVF":
            # IVF index with 100 clusters
            quantizer = faiss.IndexFlatIP(self.embedding_dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, min(100, 2))
            self.needs_training = True
        elif self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.embedding_dimension, 32)
            self.index.hnsw.efConstruction = 40
            self.needs_training = False
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        logger.info(f"Created FAISS {self.index_type} index with dimension {self.embedding_dimension}")

    def add_documents(self, embedded_chunks: List[Dict[str, Any]]) -> List[str]:
        """Add embedded documents to FAISS index.

        Args:
            embedded_chunks: List of chunks with embeddings and metadata

        Returns:
            List of document IDs added
        """
        if not embedded_chunks:
            return []

        logger.info(f"Adding {len(embedded_chunks)} documents to FAISS")

        # Prepare embeddings and metadata
        embeddings = []
        document_ids = []

        for chunk in embedded_chunks:
            embeddings.append(chunk['embedding'])
            doc_id = chunk['chunk_id']
            document_ids.append(doc_id)

            # Store document metadata
            self.documents[doc_id] = {
                'content': chunk['content'],
                'metadata': chunk['metadata'],
                'chunk_id': doc_id
            }

            # Map document ID to index position
            self.id_to_index[doc_id] = self.next_index
            self.index_to_id[self.next_index] = doc_id
            self.next_index += 1

        # Convert to numpy array
        embedding_matrix = np.array(embeddings, dtype=np.float32)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embedding_matrix)

        # Train index if needed (IVF)
        if hasattr(self, 'needs_training') and self.needs_training and not self.index.is_trained:
            if len(embedding_matrix) >= self.index.nlist:
                self.index.train(embedding_matrix)
                self.needs_training = False
                logger.info("Trained FAISS IVF index")

        # Add to index
        try:
            self.index.add(embedding_matrix)
            logger.info(f"Successfully added {len(document_ids)} documents to FAISS")

            # Save index and metadata
            self._save_index()

            return document_ids

        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {str(e)}")
            raise

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents in FAISS.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Metadata filters (applied post-search)

        Returns:
            List of similar documents with scores
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty")
            return []

        try:
            # Normalize query embedding
            query_array = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_array)

            # Search
            scores, indices = self.index.search(query_array, min(top_k, self.index.ntotal))

            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue

                doc_id = self.index_to_id.get(idx)
                if not doc_id or doc_id not in self.documents:
                    continue

                doc_data = self.documents[doc_id]

                # Apply metadata filtering
                if filter_dict:
                    metadata = doc_data['metadata']
                    if not self._matches_filter(metadata, filter_dict):
                        continue

                results.append({
                    'content': doc_data['content'],
                    'metadata': doc_data['metadata'],
                    'chunk_id': doc_data['chunk_id'],
                    'score': float(score),
                    'faiss_index': int(idx)
                })

            logger.debug(f"Found {len(results)} similar documents in FAISS")
            return results

        except Exception as e:
            logger.error(f"Error searching FAISS: {str(e)}")
            raise

    def _matches_filter(self, metadata: Dict, filter_dict: Dict) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from FAISS (marks as deleted in metadata).

        Note: FAISS doesn't support efficient deletion, so we mark documents as deleted
        and filter them out during search.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            True if successful
        """
        try:
            for doc_id in document_ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]

                if doc_id in self.id_to_index:
                    idx = self.id_to_index[doc_id]
                    del self.id_to_index[doc_id]
                    del self.index_to_id[idx]

            logger.info(f"Marked {len(document_ids)} documents as deleted in FAISS")
            self._save_index()
            return True

        except Exception as e:
            logger.error(f"Error deleting documents from FAISS: {str(e)}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the FAISS index."""
        return {
            'collection_name': self.collection_name,
            'total_documents': len(self.documents),
            'faiss_index_size': self.index.ntotal,
            'embedding_dimension': self.embedding_dimension,
            'index_type': self.index_type,
            'persist_directory': str(self.persist_directory)
        }

    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            index_path = self.persist_directory / f"{self.collection_name}.faiss"
            faiss.write_index(self.index, str(index_path))

            # Save metadata
            metadata_path = self.persist_directory / f"{self.collection_name}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'documents': self.documents,
                    'id_to_index': self.id_to_index,
                    'index_to_id': {str(k): v for k, v in self.index_to_id.items()},
                    'next_index': self.next_index,
                    'embedding_dimension': self.embedding_dimension,
                    'index_type': self.index_type
                }, f, ensure_ascii=False, indent=2)

            logger.debug("Saved FAISS index and metadata")

        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")

    def _load_index(self):
        """Load FAISS index and metadata from disk."""
        try:
            index_path = self.persist_directory / f"{self.collection_name}.faiss"
            metadata_path = self.persist_directory / f"{self.collection_name}_metadata.json"

            if index_path.exists() and metadata_path.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))

                # Load metadata
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self.documents = data.get('documents', {})
                self.id_to_index = data.get('id_to_index', {})
                self.index_to_id = {int(k): v for k, v in data.get('index_to_id', {}).items()}
                self.next_index = data.get('next_index', 0)

                logger.info(f"Loaded FAISS index with {len(self.documents)} documents")

        except Exception as e:
            logger.warning(f"Could not load existing FAISS index: {str(e)}")


class VectorStore:
    """Factory class for creating vector stores."""

    @staticmethod
    def create_store(
        store_type: str = "chroma",
        collection_name: str = "documents",
        persist_directory: str = "./data/vector_db",
        embedding_dimension: Optional[int] = None,
        **kwargs
    ) -> BaseVectorStore:
        """Create a vector store instance.

        Args:
            store_type: Type of vector store ("chroma", "faiss")
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            embedding_dimension: Dimension of embeddings
            **kwargs: Additional arguments for specific stores

        Returns:
            Vector store instance
        """
        if store_type.lower() == "chroma":
            return ChromaVectorStore(
                collection_name=collection_name,
                persist_directory=persist_directory,
                embedding_dimension=embedding_dimension
            )
        elif store_type.lower() == "faiss":
            if embedding_dimension is None:
                raise ValueError("embedding_dimension is required for FAISS")

            return FAISSVectorStore(
                embedding_dimension=embedding_dimension,
                persist_directory=persist_directory,
                collection_name=collection_name,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")