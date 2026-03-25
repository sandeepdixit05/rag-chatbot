"""Tests for embedding and document processing components."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.embeddings import DocumentProcessor, EmbeddingGenerator
from src.embeddings.document_processor import DocumentChunk


class TestDocumentProcessor:
    """Tests for DocumentProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor(
            chunk_size=100,
            chunk_overlap=20,
            splitter_type="recursive"
        )

    def test_process_document_basic(self):
        """Test basic document processing."""
        text = "This is a test document. " * 20  # Make it long enough to split
        metadata = {"source": "test"}

        chunks = self.processor.process_document(text, metadata, "test_doc")

        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.metadata["document_id"] == "test_doc" for chunk in chunks)

    def test_process_empty_document(self):
        """Test processing of empty document."""
        chunks = self.processor.process_document("", {"source": "test"}, "empty_doc")
        assert chunks == []

        chunks = self.processor.process_document("   \n\n  ", {"source": "test"}, "whitespace_doc")
        assert chunks == []

    def test_chunk_metadata_generation(self):
        """Test metadata generation for chunks."""
        text = "Short text that should be one chunk."
        metadata = {"author": "Test Author", "title": "Test Document"}

        chunks = self.processor.process_document(text, metadata, "test_doc")

        assert len(chunks) == 1
        chunk = chunks[0]

        assert chunk.metadata["document_id"] == "test_doc"
        assert chunk.metadata["author"] == "Test Author"
        assert chunk.metadata["title"] == "Test Document"
        assert chunk.metadata["chunk_index"] == 0
        assert chunk.metadata["total_chunks"] == 1
        assert "chunk_size" in chunk.metadata

    def test_document_id_generation(self):
        """Test automatic document ID generation."""
        text = "Test document content"
        chunks = self.processor.process_document(text, {})

        assert len(chunks) > 0
        assert chunks[0].metadata["document_id"] is not None
        assert len(chunks[0].metadata["document_id"]) == 32  # MD5 hash length

    def test_different_splitter_types(self):
        """Test different text splitter types."""
        text = "This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three."

        # Test recursive splitter
        processor_recursive = DocumentProcessor(chunk_size=50, splitter_type="recursive")
        chunks_recursive = processor_recursive.process_document(text, {}, "test")

        # Test character splitter
        processor_character = DocumentProcessor(chunk_size=50, splitter_type="character")
        chunks_character = processor_character.process_document(text, {}, "test")

        assert len(chunks_recursive) > 0
        assert len(chunks_character) > 0

        # Both should create chunks
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks_recursive)
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks_character)

    def test_invalid_splitter_type(self):
        """Test invalid splitter type handling."""
        with pytest.raises(ValueError, match="Unknown splitter type"):
            DocumentProcessor(splitter_type="invalid_type")

    def test_merge_chunks(self):
        """Test chunk merging functionality."""
        # Create some small chunks
        chunks = [
            DocumentChunk("Short chunk 1", {"doc": "test"}, "chunk_1"),
            DocumentChunk("Short chunk 2", {"doc": "test"}, "chunk_2"),
            DocumentChunk("This is a longer chunk that should not be merged", {"doc": "test"}, "chunk_3")
        ]

        merged = self.processor.merge_chunks(chunks, max_chunk_size=50)

        # Should have fewer chunks after merging
        assert len(merged) < len(chunks)
        assert any("merged" in chunk.chunk_id for chunk in merged)

    def test_filter_chunks(self):
        """Test chunk filtering."""
        chunks = [
            DocumentChunk("This is a good chunk with sufficient content", {"doc": "test"}, "good_chunk"),
            DocumentChunk("Short", {"doc": "test"}, "short_chunk"),
            DocumentChunk("123", {"doc": "test"}, "number_chunk"),
            DocumentChunk("!@#$%^&*()", {"doc": "test"}, "punct_chunk")
        ]

        filtered = self.processor.filter_chunks(chunks, min_length=20)

        assert len(filtered) == 1
        assert filtered[0].chunk_id == "good_chunk"

    def test_chunk_statistics(self):
        """Test chunk statistics generation."""
        chunks = [
            DocumentChunk("Content 1", {"document_id": "doc1"}, "chunk_1"),
            DocumentChunk("Content 2", {"document_id": "doc1"}, "chunk_2"),
            DocumentChunk("Content 3", {"document_id": "doc2"}, "chunk_3")
        ]

        stats = self.processor.get_chunk_statistics(chunks)

        assert stats["total_chunks"] == 3
        assert stats["unique_documents"] == 2
        assert "average_chunk_length" in stats
        assert "min_chunk_length" in stats
        assert "max_chunk_length" in stats

    def test_process_pdf_extraction(self):
        """Test processing PDF extraction results."""
        pdf_result = {
            "text": "Full document text content here",
            "metadata": {"file_name": "test.pdf", "author": "Test Author"},
            "pages": [
                {"page_number": 1, "text": "Page 1 content", "char_count": 14},
                {"page_number": 2, "text": "Page 2 content", "char_count": 14}
            ],
            "total_pages": 2,
            "extraction_method": "pdfplumber"
        }

        chunks = self.processor.process_pdf_extraction(pdf_result, preserve_page_info=True)

        assert len(chunks) > 0
        assert all(chunk.metadata["document_type"] == "pdf" for chunk in chunks)
        assert all(chunk.metadata["extraction_method"] == "pdfplumber" for chunk in chunks)


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock the sentence transformers to avoid actual model loading
        self.mock_model = Mock()
        self.mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    @patch('src.embeddings.embedding_generator.SentenceTransformer')
    def test_sentence_transformers_initialization(self, mock_st):
        """Test sentence transformers initialization."""
        mock_st.return_value = self.mock_model
        # Mock test embedding for dimension detection
        self.mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        generator = EmbeddingGenerator(
            model_name="test-model",
            provider="sentence_transformers"
        )

        assert generator.model is not None
        assert generator.embedding_dimension == 3
        mock_st.assert_called_once()

    def test_openai_provider_validation(self):
        """Test OpenAI provider validation."""
        with pytest.raises(ValueError, match="OpenAI API key required"):
            EmbeddingGenerator(
                model_name="text-embedding-ada-002",
                provider="openai"
            )

    def test_unsupported_provider(self):
        """Test unsupported provider handling."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            EmbeddingGenerator(
                model_name="test-model",
                provider="unsupported_provider"
            )

    @patch('src.embeddings.embedding_generator.SentenceTransformer')
    def test_generate_embeddings_sentence_transformers(self, mock_st):
        """Test embedding generation with sentence transformers."""
        mock_st.return_value = self.mock_model
        # Mock dimension detection
        self.mock_model.encode.side_effect = [
            np.array([[0.1, 0.2, 0.3]]),  # For dimension detection
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # For actual generation
        ]

        generator = EmbeddingGenerator(
            model_name="test-model",
            provider="sentence_transformers"
        )

        texts = ["First text", "Second text"]
        embeddings = generator.generate_embeddings(texts, show_progress=False)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 3
        assert isinstance(embeddings[0], list)

    def test_generate_embeddings_empty_list(self):
        """Test embedding generation with empty text list."""
        with patch('src.embeddings.embedding_generator.SentenceTransformer'):
            generator = EmbeddingGenerator(
                model_name="test-model",
                provider="sentence_transformers"
            )

            embeddings = generator.generate_embeddings([])
            assert embeddings == []

    @patch('src.embeddings.embedding_generator.SentenceTransformer')
    def test_embed_chunks(self, mock_st):
        """Test embedding of document chunks."""
        mock_st.return_value = self.mock_model
        # Mock responses for dimension detection and actual embedding
        self.mock_model.encode.side_effect = [
            np.array([[0.1, 0.2, 0.3]]),  # Dimension detection
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # Actual embeddings
        ]

        generator = EmbeddingGenerator(
            model_name="test-model",
            provider="sentence_transformers"
        )

        chunks = [
            DocumentChunk("Content 1", {"source": "doc1"}, "chunk_1"),
            DocumentChunk("Content 2", {"source": "doc2"}, "chunk_2")
        ]

        embedded_chunks = generator.embed_chunks(chunks, show_progress=False)

        assert len(embedded_chunks) == 2
        assert all("embedding" in chunk for chunk in embedded_chunks)
        assert all("embedding_model" in chunk for chunk in embedded_chunks)
        assert all("embedding_dimension" in chunk for chunk in embedded_chunks)

    def test_normalize_embedding(self):
        """Test embedding normalization."""
        with patch('src.embeddings.embedding_generator.SentenceTransformer'):
            generator = EmbeddingGenerator(
                model_name="test-model",
                provider="sentence_transformers"
            )

            embedding = [3.0, 4.0]  # Length 5 vector
            normalized = generator._normalize_embedding(embedding)

            # Check if normalized (length should be 1)
            length = np.linalg.norm(normalized)
            assert abs(length - 1.0) < 1e-6

    def test_compute_similarity_cosine(self):
        """Test cosine similarity computation."""
        with patch('src.embeddings.embedding_generator.SentenceTransformer'):
            generator = EmbeddingGenerator(
                model_name="test-model",
                provider="sentence_transformers"
            )

            query_embedding = [1.0, 0.0]
            doc_embeddings = [
                [1.0, 0.0],  # Same direction, similarity = 1
                [0.0, 1.0],  # Orthogonal, similarity = 0
                [-1.0, 0.0]  # Opposite direction, similarity = -1
            ]

            similarities = generator.compute_similarity(
                query_embedding, doc_embeddings, method="cosine"
            )

            assert len(similarities) == 3
            assert abs(similarities[0] - 1.0) < 1e-6
            assert abs(similarities[1] - 0.0) < 1e-6
            assert abs(similarities[2] - (-1.0)) < 1e-6

    def test_compute_similarity_dot_product(self):
        """Test dot product similarity computation."""
        with patch('src.embeddings.embedding_generator.SentenceTransformer'):
            generator = EmbeddingGenerator(
                model_name="test-model",
                provider="sentence_transformers"
            )

            query_embedding = [2.0, 3.0]
            doc_embeddings = [
                [1.0, 1.0],  # Dot product = 2*1 + 3*1 = 5
                [2.0, 0.0]   # Dot product = 2*2 + 3*0 = 4
            ]

            similarities = generator.compute_similarity(
                query_embedding, doc_embeddings, method="dot_product"
            )

            assert similarities[0] == 5.0
            assert similarities[1] == 4.0

    def test_compute_similarity_invalid_method(self):
        """Test invalid similarity method handling."""
        with patch('src.embeddings.embedding_generator.SentenceTransformer'):
            generator = EmbeddingGenerator(
                model_name="test-model",
                provider="sentence_transformers"
            )

            with pytest.raises(ValueError, match="Unsupported similarity method"):
                generator.compute_similarity([1, 0], [[0, 1]], method="invalid")

    def test_get_model_info(self):
        """Test model information retrieval."""
        with patch('src.embeddings.embedding_generator.SentenceTransformer'):
            generator = EmbeddingGenerator(
                model_name="test-model",
                provider="sentence_transformers",
                batch_size=16
            )

            info = generator.get_model_info()

            assert info["model_name"] == "test-model"
            assert info["provider"] == "sentence_transformers"
            assert info["batch_size"] == 16
            assert "device" in info
            assert "embedding_dimension" in info


if __name__ == "__main__":
    pytest.main([__file__])