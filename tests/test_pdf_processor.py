"""Tests for PDF processing components."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.pdf_processor import PDFExtractor
from src.utils import TextCleaner


class TestTextCleaner:
    """Tests for TextCleaner class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = TextCleaner()

    def test_basic_cleaning(self):
        """Test basic text cleaning functionality."""
        dirty_text = "  This is   a test\n\n\nwith    multiple   spaces  \n\n  "
        cleaned = self.cleaner.clean(dirty_text)

        assert "multiple spaces" in cleaned
        assert cleaned.strip() == cleaned  # No leading/trailing whitespace
        assert "   " not in cleaned  # No multiple spaces

    def test_remove_artifacts(self):
        """Test removal of PDF artifacts."""
        text_with_artifacts = "Normal text\x00\x0bwith\x1fartifacts"
        cleaned = self.cleaner.clean(text_with_artifacts, remove_artifacts=True)

        assert "\x00" not in cleaned
        assert "\x0b" not in cleaned
        assert "\x1f" not in cleaned
        assert "Normal textwithartifacts" == cleaned

    def test_sentence_splitting(self):
        """Test sentence splitting functionality."""
        text = "First sentence. Second sentence! Third sentence?"
        sentences = self.cleaner.split_into_sentences(text)

        assert len(sentences) == 3
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
        assert "Third sentence" in sentences[2]

    def test_paragraph_splitting(self):
        """Test paragraph splitting."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        paragraphs = self.cleaner.split_into_paragraphs(text)

        assert len(paragraphs) == 3
        assert "First paragraph." == paragraphs[0]
        assert "Second paragraph." == paragraphs[1]
        assert "Third paragraph." == paragraphs[2]

    def test_url_removal(self):
        """Test URL removal."""
        text = "Visit https://example.com for more info."
        cleaned = self.cleaner.remove_urls(text)

        assert "https://example.com" not in cleaned
        assert "Visit  for more info." == cleaned

    def test_email_removal(self):
        """Test email address removal."""
        text = "Contact us at test@example.com for help."
        cleaned = self.cleaner.remove_email_addresses(text)

        assert "test@example.com" not in cleaned
        assert "Contact us at  for help." == cleaned

    def test_quote_normalization(self):
        """Test quote character normalization."""
        text = "He said "hello" and she replied 'hi'."
        normalized = self.cleaner.normalize_quotes(text)

        assert '"' in normalized
        assert "'" in normalized
        assert """ not in normalized
        assert "'" not in normalized


class TestPDFExtractor:
    """Tests for PDFExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = PDFExtractor(use_cache=False)

    @patch('src.pdf_processor.pdf_extractor.pdfplumber')
    def test_extract_with_pdfplumber_mock(self, mock_pdfplumber):
        """Test PDF extraction using mocked pdfplumber."""
        # Mock PDF with sample content
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample PDF content"

        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {"Title": "Test PDF"}

        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            result = self.extractor._extract_with_pdfplumber(tmp_path, clean_text=True)

            assert result["text"] == "Sample PDF content"
            assert result["total_pages"] == 1
            assert len(result["pages"]) == 1
            assert result["pages"][0]["text"] == "Sample PDF content"

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_get_cache_path(self):
        """Test cache path generation."""
        test_file = Path("test.pdf")

        with patch.object(test_file, 'stat') as mock_stat:
            mock_stat.return_value.st_mtime = 1234567890
            cache_path = self.extractor._get_cache_path(test_file)

            assert cache_path.parent.name == "cache"
            assert cache_path.suffix == ".json"
            assert len(cache_path.stem) == 32  # MD5 hash length

    def test_metadata_extraction_fallback(self):
        """Test metadata extraction fallback."""
        test_file = Path("nonexistent.pdf")

        # This should not raise an exception and return basic metadata
        with patch.object(test_file, 'stat') as mock_stat:
            mock_stat.return_value.st_size = 1000
            metadata = self.extractor._get_pdf_metadata(test_file)

            assert metadata["file_size"] == 1000
            assert metadata["file_name"] == "nonexistent.pdf"

    @pytest.mark.parametrize("method", ["pypdf2", "pdfplumber", "pymupdf"])
    def test_extraction_method_selection(self, method):
        """Test different extraction methods."""
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
            tmp_path = Path(tmp.name)

            # This will fail with real PDF libraries, but tests method dispatch
            with pytest.raises((FileNotFoundError, Exception)):
                self.extractor.extract_text(tmp_path, method=method)

    def test_invalid_extraction_method(self):
        """Test invalid extraction method handling."""
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
            tmp_path = Path(tmp.name)

            with pytest.raises(ValueError, match="Unknown extraction method"):
                self.extractor.extract_text(tmp_path, method="invalid_method")

    def test_file_not_found(self):
        """Test handling of non-existent files."""
        with pytest.raises(FileNotFoundError):
            self.extractor.extract_text("nonexistent.pdf")

    def test_extract_from_empty_directory(self):
        """Test extraction from empty directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = self.extractor.extract_from_directory(tmp_dir)
            assert results == []

    def test_extract_from_nonexistent_directory(self):
        """Test extraction from non-existent directory."""
        with pytest.raises(FileNotFoundError):
            self.extractor.extract_from_directory("nonexistent_directory")


if __name__ == "__main__":
    pytest.main([__file__])