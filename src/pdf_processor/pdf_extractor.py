"""PDF text extraction utilities using multiple PDF processing libraries."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import hashlib

try:
    import PyPDF2
    import pdfplumber
    import fitz  # PyMuPDF
except ImportError as e:
    raise ImportError(
        "PDF processing libraries not installed. "
        "Please install: pip install PyPDF2 pdfplumber pymupdf"
    ) from e

from ..utils.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract and process text from PDF documents using multiple methods."""

    def __init__(self, use_cache: bool = True):
        """Initialize PDF extractor.

        Args:
            use_cache: Whether to cache extraction results
        """
        self.use_cache = use_cache
        self.text_cleaner = TextCleaner()

    def extract_text(
        self,
        pdf_path: Union[str, Path],
        method: str = "auto",
        clean_text: bool = True
    ) -> Dict[str, Union[str, List[Dict], Dict]]:
        """Extract text from PDF using specified method.

        Args:
            pdf_path: Path to PDF file
            method: Extraction method ("pypdf2", "pdfplumber", "pymupdf", "auto")
            clean_text: Whether to clean extracted text

        Returns:
            Dictionary containing extracted text, metadata, and page information
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Check cache first
        if self.use_cache:
            cached_result = self._load_from_cache(pdf_path)
            if cached_result:
                logger.info(f"Loaded cached extraction for {pdf_path.name}")
                return cached_result

        logger.info(f"Extracting text from {pdf_path.name} using method: {method}")

        # Try different extraction methods based on preference
        if method == "auto":
            result = self._extract_auto(pdf_path, clean_text)
        elif method == "pdfplumber":
            result = self._extract_with_pdfplumber(pdf_path, clean_text)
        elif method == "pypdf2":
            result = self._extract_with_pypdf2(pdf_path, clean_text)
        elif method == "pymupdf":
            result = self._extract_with_pymupdf(pdf_path, clean_text)
        else:
            raise ValueError(f"Unknown extraction method: {method}")

        # Add metadata
        result["metadata"] = self._get_pdf_metadata(pdf_path)
        result["extraction_method"] = method
        result["file_path"] = str(pdf_path)

        # Cache result
        if self.use_cache:
            self._save_to_cache(pdf_path, result)

        logger.info(f"Successfully extracted text from {pdf_path.name}")
        return result

    def _extract_auto(self, pdf_path: Path, clean_text: bool) -> Dict:
        """Try different extraction methods automatically."""
        methods = ["pdfplumber", "pymupdf", "pypdf2"]

        for method in methods:
            try:
                logger.debug(f"Trying {method} for {pdf_path.name}")
                if method == "pdfplumber":
                    result = self._extract_with_pdfplumber(pdf_path, clean_text)
                elif method == "pymupdf":
                    result = self._extract_with_pymupdf(pdf_path, clean_text)
                elif method == "pypdf2":
                    result = self._extract_with_pypdf2(pdf_path, clean_text)

                # Check if extraction was successful
                if result["text"] and len(result["text"].strip()) > 50:
                    logger.info(f"Successfully extracted with {method}")
                    return result

            except Exception as e:
                logger.warning(f"Method {method} failed: {str(e)}")
                continue

        raise RuntimeError(f"All extraction methods failed for {pdf_path.name}")

    def _extract_with_pdfplumber(self, pdf_path: Path, clean_text: bool) -> Dict:
        """Extract text using pdfplumber."""
        with pdfplumber.open(pdf_path) as pdf:
            pages_data = []
            full_text = []

            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""

                if clean_text:
                    text = self.text_cleaner.clean(text)

                pages_data.append({
                    "page_number": page_num,
                    "text": text,
                    "char_count": len(text)
                })
                full_text.append(text)

        return {
            "text": "\n\n".join(full_text),
            "pages": pages_data,
            "total_pages": len(pages_data)
        }

    def _extract_with_pypdf2(self, pdf_path: Path, clean_text: bool) -> Dict:
        """Extract text using PyPDF2."""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            pages_data = []
            full_text = []

            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()

                if clean_text:
                    text = self.text_cleaner.clean(text)

                pages_data.append({
                    "page_number": page_num,
                    "text": text,
                    "char_count": len(text)
                })
                full_text.append(text)

        return {
            "text": "\n\n".join(full_text),
            "pages": pages_data,
            "total_pages": len(pages_data)
        }

    def _extract_with_pymupdf(self, pdf_path: Path, clean_text: bool) -> Dict:
        """Extract text using PyMuPDF."""
        doc = fitz.open(pdf_path)
        pages_data = []
        full_text = []

        try:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()

                if clean_text:
                    text = self.text_cleaner.clean(text)

                pages_data.append({
                    "page_number": page_num + 1,
                    "text": text,
                    "char_count": len(text)
                })
                full_text.append(text)
        finally:
            doc.close()

        return {
            "text": "\n\n".join(full_text),
            "pages": pages_data,
            "total_pages": len(pages_data)
        }

    def _get_pdf_metadata(self, pdf_path: Path) -> Dict:
        """Extract PDF metadata."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                metadata = pdf.metadata or {}
                return {
                    "title": metadata.get("Title", ""),
                    "author": metadata.get("Author", ""),
                    "subject": metadata.get("Subject", ""),
                    "creator": metadata.get("Creator", ""),
                    "producer": metadata.get("Producer", ""),
                    "creation_date": str(metadata.get("CreationDate", "")),
                    "modification_date": str(metadata.get("ModDate", "")),
                    "file_size": pdf_path.stat().st_size,
                    "file_name": pdf_path.name
                }
        except Exception as e:
            logger.warning(f"Could not extract metadata: {str(e)}")
            return {
                "file_size": pdf_path.stat().st_size,
                "file_name": pdf_path.name
            }

    def _get_cache_path(self, pdf_path: Path) -> Path:
        """Get cache file path for PDF."""
        # Create hash of file path and modification time for cache key
        cache_key = hashlib.md5(
            f"{pdf_path.absolute()}_{pdf_path.stat().st_mtime}".encode()
        ).hexdigest()

        cache_dir = Path("./data/processed/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        return cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, pdf_path: Path) -> Optional[Dict]:
        """Load cached extraction result."""
        try:
            cache_path = self._get_cache_path(pdf_path)
            if cache_path.exists():
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load cache: {str(e)}")
        return None

    def _save_to_cache(self, pdf_path: Path, result: Dict) -> None:
        """Save extraction result to cache."""
        try:
            cache_path = self._get_cache_path(pdf_path)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Could not save to cache: {str(e)}")

    def extract_from_directory(
        self,
        directory_path: Union[str, Path],
        method: str = "auto",
        clean_text: bool = True
    ) -> List[Dict]:
        """Extract text from all PDFs in a directory.

        Args:
            directory_path: Path to directory containing PDFs
            method: Extraction method to use
            clean_text: Whether to clean extracted text

        Returns:
            List of extraction results for each PDF
        """
        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        pdf_files = list(directory_path.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return []

        results = []
        for pdf_file in pdf_files:
            try:
                result = self.extract_text(pdf_file, method, clean_text)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to extract from {pdf_file.name}: {str(e)}")

        logger.info(f"Extracted text from {len(results)} PDFs")
        return results