"""Document retrieval system with advanced filtering and ranking."""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import re
from collections import Counter

from .vector_store import BaseVectorStore
from ..embeddings.embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a document retrieval result."""

    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    score: float
    rank: int
    source_info: Optional[Dict] = None


class DocumentRetriever:
    """Advanced document retrieval with re-ranking and filtering."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_generator: EmbeddingGenerator,
        default_top_k: int = 10,
        score_threshold: float = 0.0,
        enable_reranking: bool = True
    ):
        """Initialize document retriever.

        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator for query encoding
            default_top_k: Default number of documents to retrieve
            score_threshold: Minimum similarity score threshold
            enable_reranking: Whether to enable advanced re-ranking
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.default_top_k = default_top_k
        self.score_threshold = score_threshold
        self.enable_reranking = enable_reranking

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict] = None,
        rerank_query: bool = None,
        include_metadata_fields: Optional[List[str]] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filter_dict: Metadata filters
            rerank_query: Whether to apply re-ranking (overrides default)
            include_metadata_fields: Specific metadata fields to include

        Returns:
            List of retrieval results
        """
        top_k = top_k or self.default_top_k
        rerank_query = rerank_query if rerank_query is not None else self.enable_reranking

        logger.info(f"Retrieving documents for query: '{query[:50]}...'")

        try:
            # Generate query embedding
            query_embeddings = self.embedding_generator.generate_embeddings(
                [query],
                show_progress=False
            )
            query_embedding = query_embeddings[0]

            # Retrieve initial candidates (get more if re-ranking)
            initial_k = top_k * 2 if rerank_query else top_k

            raw_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=initial_k,
                filter_dict=filter_dict
            )

            # Filter by score threshold
            filtered_results = [
                result for result in raw_results
                if result.get('score', 0) >= self.score_threshold
            ]

            logger.debug(f"Found {len(raw_results)} candidates, {len(filtered_results)} above threshold")

            # Apply re-ranking if enabled
            if rerank_query and filtered_results:
                filtered_results = self._rerank_results(query, filtered_results)

            # Limit to requested number
            final_results = filtered_results[:top_k]

            # Convert to RetrievalResult objects
            retrieval_results = []
            for i, result in enumerate(final_results):
                # Filter metadata fields if specified
                metadata = result['metadata']
                if include_metadata_fields:
                    metadata = {
                        k: v for k, v in metadata.items()
                        if k in include_metadata_fields
                    }

                retrieval_results.append(RetrievalResult(
                    content=result['content'],
                    metadata=metadata,
                    chunk_id=result['chunk_id'],
                    score=result['score'],
                    rank=i + 1,
                    source_info=self._extract_source_info(result['metadata'])
                ))

            logger.info(f"Retrieved {len(retrieval_results)} relevant documents")
            return retrieval_results

        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise

    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Re-rank results using additional relevance signals.

        Args:
            query: Original query
            results: Initial retrieval results

        Returns:
            Re-ranked results
        """
        logger.debug("Applying re-ranking to results")

        # Extract query keywords
        query_keywords = self._extract_keywords(query.lower())

        # Calculate additional relevance scores
        for result in results:
            content = result['content'].lower()

            # Keyword matching score
            keyword_score = self._calculate_keyword_score(content, query_keywords)

            # Document quality score
            quality_score = self._calculate_quality_score(result['content'], result['metadata'])

            # Recency score (if date information available)
            recency_score = self._calculate_recency_score(result['metadata'])

            # Source authority score
            authority_score = self._calculate_authority_score(result['metadata'])

            # Combine scores (weighted)
            original_score = result['score']
            combined_score = (
                original_score * 0.5 +      # Vector similarity (50%)
                keyword_score * 0.2 +       # Keyword matching (20%)
                quality_score * 0.15 +      # Content quality (15%)
                recency_score * 0.1 +       # Recency (10%)
                authority_score * 0.05      # Source authority (5%)
            )

            result['rerank_score'] = combined_score
            result['score_components'] = {
                'vector_similarity': original_score,
                'keyword_matching': keyword_score,
                'quality_score': quality_score,
                'recency_score': recency_score,
                'authority_score': authority_score
            }

        # Sort by combined score
        results.sort(key=lambda x: x['rerank_score'], reverse=True)

        logger.debug("Re-ranking completed")
        return results

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Simple keyword extraction (can be improved with NLP)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)

        # Filter out common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'among', 'this', 'that', 'these', 'those', 'what', 'which',
            'who', 'when', 'where', 'why', 'how', 'can', 'could', 'would', 'should', 'may',
            'might', 'must', 'shall', 'will', 'have', 'has', 'had', 'been', 'being'
        }

        keywords = [word for word in words if word.lower() not in stop_words]
        return keywords

    def _calculate_keyword_score(self, content: str, query_keywords: List[str]) -> float:
        """Calculate keyword matching score."""
        if not query_keywords:
            return 0.0

        content_words = content.split()
        matches = sum(1 for keyword in query_keywords if keyword in content)

        return matches / len(query_keywords)

    def _calculate_quality_score(self, content: str, metadata: Dict) -> float:
        """Calculate content quality score based on various factors."""
        score = 0.0

        # Length factor (moderate length is better)
        content_length = len(content)
        if 100 <= content_length <= 2000:
            score += 0.3
        elif content_length < 100:
            score += 0.1
        else:
            score += 0.2

        # Sentence structure (presence of proper punctuation)
        sentences = len(re.findall(r'[.!?]+', content))
        words = len(content.split())
        if words > 0:
            sentence_ratio = sentences / (words / 15)  # ~15 words per sentence
            if 0.5 <= sentence_ratio <= 2.0:
                score += 0.3

        # Capitalization (proper capitalization suggests quality)
        if re.search(r'^[A-Z][a-z]', content):
            score += 0.2

        # Metadata quality indicators
        if metadata.get('title') or metadata.get('author'):
            score += 0.2

        return min(score, 1.0)

    def _calculate_recency_score(self, metadata: Dict) -> float:
        """Calculate recency score based on document date."""
        # Simple implementation - can be enhanced with actual date parsing
        creation_date = metadata.get('creation_date', '')
        modification_date = metadata.get('modification_date', '')

        # If we have date information, newer documents get higher scores
        # This is a placeholder - implement actual date scoring logic
        if creation_date or modification_date:
            return 0.5  # Neutral score for now

        return 0.3  # Lower score for documents without date info

    def _calculate_authority_score(self, metadata: Dict) -> float:
        """Calculate source authority score."""
        score = 0.5  # Base score

        # Check for authority indicators
        file_name = metadata.get('file_name', '').lower()

        # Academic or official sources
        if any(keyword in file_name for keyword in ['official', 'policy', 'guide', 'manual']):
            score += 0.3

        # Technical documentation
        if any(keyword in file_name for keyword in ['technical', 'spec', 'documentation']):
            score += 0.2

        return min(score, 1.0)

    def _extract_source_info(self, metadata: Dict) -> Dict[str, Any]:
        """Extract relevant source information from metadata."""
        return {
            'file_name': metadata.get('file_name', 'Unknown'),
            'page_number': metadata.get('page_number'),
            'document_type': metadata.get('document_type', 'pdf'),
            'author': metadata.get('author', ''),
            'title': metadata.get('title', '')
        }

    def retrieve_with_context(
        self,
        query: str,
        context_window: int = 1,
        **kwargs
    ) -> List[RetrievalResult]:
        """Retrieve documents with surrounding context chunks.

        Args:
            query: Search query
            context_window: Number of chunks before/after to include
            **kwargs: Additional arguments for retrieve method

        Returns:
            List of retrieval results with context
        """
        # Get initial results
        results = self.retrieve(query, **kwargs)

        if not results or context_window <= 0:
            return results

        # For each result, try to find surrounding chunks
        enhanced_results = []

        for result in results:
            # Try to find context chunks (this requires chunks to have sequence info)
            chunk_metadata = result.metadata
            document_id = chunk_metadata.get('document_id')
            chunk_index = chunk_metadata.get('chunk_index')

            if document_id and chunk_index is not None:
                # Find surrounding chunks
                context_chunks = self._find_context_chunks(
                    document_id, chunk_index, context_window
                )

                if context_chunks:
                    # Combine content with context
                    full_content = self._combine_with_context(result.content, context_chunks)
                    result.content = full_content
                    result.metadata['has_context'] = True
                    result.metadata['context_chunks'] = len(context_chunks)

            enhanced_results.append(result)

        return enhanced_results

    def _find_context_chunks(
        self,
        document_id: str,
        center_chunk_index: int,
        context_window: int
    ) -> List[str]:
        """Find context chunks around a central chunk."""
        # This would require querying the vector store for chunks from the same document
        # with adjacent chunk indices. Implementation depends on metadata structure.

        # Placeholder implementation
        context_chunks = []

        # In a real implementation, you would:
        # 1. Query vector store for chunks with same document_id
        # 2. Filter for chunk_index in range [center_chunk_index - context_window, center_chunk_index + context_window]
        # 3. Sort by chunk_index
        # 4. Extract content

        return context_chunks

    def _combine_with_context(self, main_content: str, context_chunks: List[str]) -> str:
        """Combine main content with context chunks."""
        if not context_chunks:
            return main_content

        # Simple combination - can be enhanced
        return "\n\n".join(context_chunks + [f"[MAIN CONTENT]\n{main_content}"])

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics."""
        vector_stats = self.vector_store.get_collection_stats()
        embedding_stats = self.embedding_generator.get_model_info()

        return {
            'vector_store': vector_stats,
            'embedding_model': embedding_stats,
            'retrieval_config': {
                'default_top_k': self.default_top_k,
                'score_threshold': self.score_threshold,
                'enable_reranking': self.enable_reranking
            }
        }

    def update_settings(
        self,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        enable_reranking: Optional[bool] = None
    ):
        """Update retrieval settings."""
        if top_k is not None:
            self.default_top_k = top_k
        if score_threshold is not None:
            self.score_threshold = score_threshold
        if enable_reranking is not None:
            self.enable_reranking = enable_reranking

        logger.info("Updated retrieval settings")