"""Text cleaning and preprocessing utilities."""

import re
from typing import List, Optional


class TextCleaner:
    """Clean and preprocess extracted text for better processing."""

    def __init__(self):
        """Initialize text cleaner with regex patterns."""
        # Common patterns to clean
        self.patterns = {
            # Multiple consecutive whitespace
            'multiple_spaces': re.compile(r'\s+'),
            # Page breaks and form feeds
            'page_breaks': re.compile(r'[\f\r]+'),
            # Multiple consecutive newlines
            'multiple_newlines': re.compile(r'\n{3,}'),
            # Leading/trailing whitespace on lines
            'line_whitespace': re.compile(r'^\s+|\s+$', re.MULTILINE),
            # Common PDF artifacts
            'pdf_artifacts': re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]'),
            # Headers/footers (simple heuristic - lines that appear frequently)
            'headers_footers': re.compile(r'^(page\s*\d+|chapter\s*\d+|\d+\s*$)', re.IGNORECASE | re.MULTILINE),
        }

    def clean(
        self,
        text: str,
        remove_headers_footers: bool = True,
        normalize_whitespace: bool = True,
        remove_artifacts: bool = True,
        min_line_length: int = 3
    ) -> str:
        """Clean and normalize text.

        Args:
            text: Raw text to clean
            remove_headers_footers: Remove common header/footer patterns
            normalize_whitespace: Normalize whitespace characters
            remove_artifacts: Remove PDF artifacts and control characters
            min_line_length: Minimum length for lines to keep

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove PDF artifacts and control characters
        if remove_artifacts:
            text = self.patterns['pdf_artifacts'].sub('', text)

        # Normalize line breaks
        text = self.patterns['page_breaks'].sub('\n', text)

        # Remove headers/footers (simple approach)
        if remove_headers_footers:
            text = self.patterns['headers_footers'].sub('', text)

        # Split into lines and clean each
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Remove leading/trailing whitespace
            line = line.strip()

            # Skip very short lines (likely artifacts)
            if len(line) < min_line_length:
                continue

            # Skip lines that are mostly numbers (page numbers, etc.)
            if re.match(r'^\d+\s*$', line):
                continue

            cleaned_lines.append(line)

        # Rejoin lines
        text = '\n'.join(cleaned_lines)

        # Normalize whitespace
        if normalize_whitespace:
            # Replace multiple spaces with single space
            text = self.patterns['multiple_spaces'].sub(' ', text)

            # Replace multiple newlines with double newlines
            text = self.patterns['multiple_newlines'].sub('\n\n', text)

        return text.strip()

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with NLTK/spaCy)
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs.

        Args:
            text: Text to split

        Returns:
            List of paragraphs
        """
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]

    def remove_urls(self, text: str) -> str:
        """Remove URLs from text.

        Args:
            text: Text containing URLs

        Returns:
            Text with URLs removed
        """
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        return url_pattern.sub('', text)

    def remove_email_addresses(self, text: str) -> str:
        """Remove email addresses from text.

        Args:
            text: Text containing email addresses

        Returns:
            Text with email addresses removed
        """
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        return email_pattern.sub('', text)

    def normalize_quotes(self, text: str) -> str:
        """Normalize quote characters.

        Args:
            text: Text with various quote characters

        Returns:
            Text with normalized quotes
        """
        # Replace curly quotes with straight quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text

    def clean_for_embedding(self, text: str) -> str:
        """Clean text specifically for embedding generation.

        Args:
            text: Raw text

        Returns:
            Text optimized for embedding
        """
        # Apply standard cleaning
        text = self.clean(text)

        # Additional cleaning for embeddings
        text = self.remove_urls(text)
        text = self.remove_email_addresses(text)
        text = self.normalize_quotes(text)

        # Remove excessive punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\"\']+', ' ', text)

        # Final whitespace normalization
        text = self.patterns['multiple_spaces'].sub(' ', text)

        return text.strip()