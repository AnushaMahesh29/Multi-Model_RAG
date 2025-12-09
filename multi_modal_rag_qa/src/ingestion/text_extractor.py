"""
Text Extractor Module
Extracts text content from PDF pages using PyMuPDF.
"""

import fitz  # PyMuPDF
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextExtractor:
    """
    Extracts text from PDF pages.
    """

    def __init__(self, doc: fitz.Document):
        """
        Initialize the text extractor.

        Args:
            doc: PyMuPDF Document object
        """
        self.doc = doc

    def extract_text_from_page(self, page_num: int) -> Dict[str, Any]:
        """
        Extract text from a single page.

        Args:
            page_num: Page number (0-indexed)

        Returns:
            Dictionary with page number and extracted text
        """
        try:
            page = self.doc[page_num]
            text = page.get_text("text")

            result = {
                "page_num": page_num,
                "text": text.strip(),
                "char_count": len(text.strip()),
            }

            logger.info(
                f"Extracted {result['char_count']} characters from page {page_num}"
            )
            return result

        except Exception as e:
            logger.error(f"Error extracting text from page {page_num}: {e}")
            return {"page_num": page_num, "text": "", "char_count": 0, "error": str(e)}

    def extract_text_from_all_pages(self) -> List[Dict[str, Any]]:
        """
        Extract text from all pages in the document.

        Returns:
            List of dictionaries containing text from each page
        """
        all_text = []

        for page_num in range(len(self.doc)):
            page_text = self.extract_text_from_page(page_num)
            all_text.append(page_text)

        logger.info(f"Extracted text from {len(all_text)} pages")
        return all_text

    def extract_text_with_blocks(self, page_num: int) -> Dict[str, Any]:
        """
        Extract text with block information (useful for layout analysis).

        Args:
            page_num: Page number (0-indexed)

        Returns:
            Dictionary with page number and text blocks
        """
        try:
            page = self.doc[page_num]
            blocks = page.get_text("blocks")

            text_blocks = []
            for block in blocks:
                if block[6] == 0:  # block type 0 is text
                    text_blocks.append(
                        {
                            "bbox": block[:4],  # bounding box coordinates
                            "text": block[4].strip(),
                            "block_no": block[5],
                        }
                    )

            result = {
                "page_num": page_num,
                "blocks": text_blocks,
                "block_count": len(text_blocks),
            }

            logger.info(f"Extracted {len(text_blocks)} text blocks from page {page_num}")
            return result

        except Exception as e:
            logger.error(f"Error extracting text blocks from page {page_num}: {e}")
            return {
                "page_num": page_num,
                "blocks": [],
                "block_count": 0,
                "error": str(e),
            }


if __name__ == "__main__":
    # Example usage
    from pdf_loader import PDFLoader

    pdf_path = "data/raw/qatar_test_doc.pdf"

    with PDFLoader(pdf_path) as loader:
        extractor = TextExtractor(loader.doc)

        # Extract text from first page
        page_0_text = extractor.extract_text_from_page(0)
        print(f"Page 0 text preview: {page_0_text['text'][:200]}...")

        # Extract all text
        all_text = extractor.extract_text_from_all_pages()
        print(f"\nTotal pages with text: {len(all_text)}")

        # Extract with blocks
        blocks = extractor.extract_text_with_blocks(0)
        print(f"\nText blocks on page 0: {blocks['block_count']}")
