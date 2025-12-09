"""
PDF Loader Module
Loads PDF documents using PyMuPDF (fitz) and provides basic document information.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFLoader:
    """
    Loads and provides access to PDF documents.
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize the PDF loader.
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = Path(pdf_path)
        self.doc: Optional[fitz.Document] = None
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    def load(self) -> fitz.Document:
        """
        Load the PDF document.
        
        Returns:
            fitz.Document object
        """
        try:
            self.doc = fitz.open(self.pdf_path)
            logger.info(f"Successfully loaded PDF: {self.pdf_path.name}")
            logger.info(f"Total pages: {len(self.doc)}")
            return self.doc
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get PDF metadata.
        
        Returns:
            Dictionary containing metadata
        """
        if not self.doc:
            raise ValueError("PDF not loaded. Call load() first.")
        
        metadata = {
            "filename": self.pdf_path.name,
            "page_count": len(self.doc),
            "title": self.doc.metadata.get("title", ""),
            "author": self.doc.metadata.get("author", ""),
            "subject": self.doc.metadata.get("subject", ""),
            "creator": self.doc.metadata.get("creator", ""),
        }
        
        return metadata
    
    def get_page(self, page_num: int) -> fitz.Page:
        """
        Get a specific page from the PDF.
        
        Args:
            page_num: Page number (0-indexed)
            
        Returns:
            fitz.Page object
        """
        if not self.doc:
            raise ValueError("PDF not loaded. Call load() first.")
        
        if page_num < 0 or page_num >= len(self.doc):
            raise ValueError(f"Invalid page number: {page_num}")
        
        return self.doc[page_num]
    
    def close(self):
        """
        Close the PDF document.
        """
        if self.doc:
            self.doc.close()
            logger.info(f"Closed PDF: {self.pdf_path.name}")
    
    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


if __name__ == "__main__":
    # Example usage
    pdf_path = "data/raw/qatar_test_doc.pdf"
    
    with PDFLoader(pdf_path) as loader:
        metadata = loader.get_metadata()
        print("PDF Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        # Access first page
        if metadata["page_count"] > 0:
            page = loader.get_page(0)
            print(f"\nFirst page dimensions: {page.rect.width} x {page.rect.height}")
