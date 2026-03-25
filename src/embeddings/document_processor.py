"""Document chunking and preprocessing for RAG pipeline."""

import logging
from typing import Dict, List, Optional, Tuple, Union
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""

    content: str
    metadata: Dict
    chunk_id: str
    start_index: int = 0
    end_index: int = 0


class DocumentProcessor:
    """Process documents into chunks suitable for embedding and retrieval."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        splitter_type: str = "recursive",
        separators: Optional[List[str]] = None
    ):
        """Initialize document processor.

        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between consecutive chunks
            splitter_type: Type of text splitter ("recursive", "character", "token")
            separators: Custom separators for recursive splitter
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter_type = splitter_type

        # Initialize text splitter
        self.text_splitter = self._create_text_splitter(separators)

    def _create_text_splitter(self, separators: Optional[List[str]]):
        """Create appropriate text splitter based on configuration."""
        if self.splitter_type == "recursive":
            separators = separators or [
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence endings
                " ",     # Word boundaries
                ""       # Character-level split as last resort
            ]
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=separators,
                length_function=len,
                add_start_index=True
            )
        elif self.splitter_type == "character":
            return CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n\n",
                length_function=len
            )
        elif self.splitter_type == "token":
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            raise ValueError(f"Unknown splitter type: {self.splitter_type}")

    def process_document(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        document_id: Optional[str] = None
    ) -> List[DocumentChunk]:
        """Process a single document into chunks.

        Args:
            text: Document text to process
            metadata: Additional metadata for the document
            document_id: Unique identifier for the document

        Returns:
            List of document chunks
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        metadata = metadata or {}

        # Generate document ID if not provided
        if not document_id:
            document_id = hashlib.md5(text.encode()).hexdigest()

        logger.info(f"Processing document {document_id} ({len(text)} characters)")

        # Split text into chunks
        try:
            if self.splitter_type in ["recursive", "character"]:
                # These splitters support start_index
                text_chunks = self.text_splitter.create_documents(
                    [text],
                    metadatas=[metadata]
                )
            else:
                # Token splitter doesn't support metadata
                text_chunks = self.text_splitter.split_text(text)
                text_chunks = [
                    type('Document', (), {
                        'page_content': chunk,
                        'metadata': metadata.copy()
                    })() for chunk in text_chunks
                ]

        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}")
            # Fallback: create single chunk
            return [DocumentChunk(
                content=text,
                metadata=metadata,
                chunk_id=f"{document_id}_0"
            )]

        # Convert to DocumentChunk objects
        chunks = []
        for i, chunk in enumerate(text_chunks):
            chunk_content = chunk.page_content
            chunk_metadata = chunk.metadata.copy()

            # Add chunk-specific metadata
            chunk_metadata.update({
                'document_id': document_id,
                'chunk_index': i,
                'chunk_size': len(chunk_content),
                'total_chunks': len(text_chunks)
            })

            # Generate unique chunk ID
            chunk_id = f"{document_id}_{i}"

            # Get start/end indices if available
            start_index = chunk_metadata.get('start_index', 0)
            end_index = start_index + len(chunk_content)

            chunks.append(DocumentChunk(
                content=chunk_content,
                metadata=chunk_metadata,
                chunk_id=chunk_id,
                start_index=start_index,
                end_index=end_index
            ))

        logger.info(f"Created {len(chunks)} chunks from document {document_id}")
        return chunks

    def process_pdf_extraction(
        self,
        pdf_extraction_result: Dict,
        preserve_page_info: bool = True
    ) -> List[DocumentChunk]:
        """Process PDF extraction result into chunks.

        Args:
            pdf_extraction_result: Result from PDFExtractor
            preserve_page_info: Whether to preserve page information in metadata

        Returns:
            List of document chunks
        """
        text = pdf_extraction_result.get('text', '')
        metadata = pdf_extraction_result.get('metadata', {})
        pages = pdf_extraction_result.get('pages', [])

        # Add PDF-specific metadata
        metadata.update({
            'total_pages': pdf_extraction_result.get('total_pages', 0),
            'extraction_method': pdf_extraction_result.get('extraction_method', 'unknown'),
            'document_type': 'pdf'
        })

        if preserve_page_info and pages:
            # Process each page separately to maintain page boundaries
            all_chunks = []

            for page_data in pages:
                page_text = page_data.get('text', '')
                if not page_text.strip():
                    continue

                page_metadata = metadata.copy()
                page_metadata.update({
                    'page_number': page_data.get('page_number'),
                    'page_char_count': page_data.get('char_count', len(page_text))
                })

                # Generate page-specific document ID
                base_id = metadata.get('file_name', 'unknown')
                page_doc_id = f"{base_id}_page_{page_data.get('page_number')}"

                page_chunks = self.process_document(
                    page_text,
                    page_metadata,
                    page_doc_id
                )
                all_chunks.extend(page_chunks)

            return all_chunks
        else:
            # Process entire document as one
            document_id = metadata.get('file_name', 'unknown')
            return self.process_document(text, metadata, document_id)

    def merge_chunks(
        self,
        chunks: List[DocumentChunk],
        max_chunk_size: Optional[int] = None
    ) -> List[DocumentChunk]:
        """Merge small chunks to optimize embedding efficiency.

        Args:
            chunks: List of chunks to potentially merge
            max_chunk_size: Maximum size for merged chunks

        Returns:
            List of potentially merged chunks
        """
        if not chunks:
            return []

        max_size = max_chunk_size or self.chunk_size
        merged_chunks = []
        current_chunk = None

        for chunk in chunks:
            if current_chunk is None:
                current_chunk = chunk
            elif (len(current_chunk.content) + len(chunk.content) + 1) <= max_size:
                # Merge chunks
                merged_content = current_chunk.content + "\n" + chunk.content
                merged_metadata = current_chunk.metadata.copy()
                merged_metadata.update({
                    'merged_chunk_ids': [current_chunk.chunk_id, chunk.chunk_id],
                    'chunk_size': len(merged_content)
                })

                current_chunk = DocumentChunk(
                    content=merged_content,
                    metadata=merged_metadata,
                    chunk_id=f"{current_chunk.chunk_id}_merged",
                    start_index=current_chunk.start_index,
                    end_index=chunk.end_index
                )
            else:
                # Add current chunk and start new one
                merged_chunks.append(current_chunk)
                current_chunk = chunk

        # Add the last chunk
        if current_chunk:
            merged_chunks.append(current_chunk)

        logger.info(f"Merged {len(chunks)} chunks into {len(merged_chunks)} chunks")
        return merged_chunks

    def filter_chunks(
        self,
        chunks: List[DocumentChunk],
        min_length: int = 50,
        max_length: Optional[int] = None,
        remove_short_sentences: bool = True
    ) -> List[DocumentChunk]:
        """Filter chunks based on length and content quality.

        Args:
            chunks: List of chunks to filter
            min_length: Minimum chunk length
            max_length: Maximum chunk length (optional)
            remove_short_sentences: Remove chunks that are too short to be meaningful

        Returns:
            Filtered list of chunks
        """
        filtered_chunks = []

        for chunk in chunks:
            content = chunk.content.strip()

            # Skip empty chunks
            if not content:
                continue

            # Check length constraints
            if len(content) < min_length:
                continue

            if max_length and len(content) > max_length:
                continue

            # Skip chunks that are just numbers or single words
            if remove_short_sentences:
                if len(content.split()) < 3:
                    continue

                # Skip chunks that are mostly punctuation
                if len(re.sub(r'[^\w\s]', '', content)) < min_length // 2:
                    continue

            filtered_chunks.append(chunk)

        logger.info(f"Filtered {len(chunks)} chunks to {len(filtered_chunks)} chunks")
        return filtered_chunks

    def get_chunk_statistics(self, chunks: List[DocumentChunk]) -> Dict:
        """Get statistics about the chunks.

        Args:
            chunks: List of chunks to analyze

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {"total_chunks": 0}

        chunk_lengths = [len(chunk.content) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk_lengths),
            "average_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "unique_documents": len(set(
                chunk.metadata.get('document_id', '') for chunk in chunks
            )),
            "chunks_per_document": len(chunks) / max(1, len(set(
                chunk.metadata.get('document_id', '') for chunk in chunks
            )))
        }