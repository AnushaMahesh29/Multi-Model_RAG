"""
Table Extractor Module
Extracts tables from PDF pages using Camelot.
"""

import camelot
from typing import List, Dict, Any
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TableExtractor:
    """
    Extracts tables from PDF documents using Camelot.
    """

    def __init__(self, pdf_path: str):
        """
        Initialize the table extractor.

        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    def extract_tables_from_page(
        self, page_num: int, flavor: str = "lattice"
    ) -> Dict[str, Any]:
        """
        Extract tables from a single page.

        Args:
            page_num: Page number (1-indexed for Camelot)
            flavor: 'lattice' for tables with lines, 'stream' for tables without lines

        Returns:
            Dictionary with page number and extracted tables
        """
        try:
            # Camelot uses 1-indexed page numbers
            tables = camelot.read_pdf(
                str(self.pdf_path), pages=str(page_num), flavor=flavor
            )

            extracted_tables = []
            for idx, table in enumerate(tables):
                extracted_tables.append(
                    {
                        "table_num": idx,
                        "dataframe": table.df,
                        "shape": table.df.shape,
                        "accuracy": table.accuracy,
                        "whitespace": table.whitespace,
                    }
                )

            result = {
                "page_num": page_num - 1,  # Convert back to 0-indexed
                "tables": extracted_tables,
                "table_count": len(extracted_tables),
            }

            logger.info(f"Extracted {len(extracted_tables)} tables from page {page_num}")
            return result

        except Exception as e:
            logger.error(f"Error extracting tables from page {page_num}: {e}")
            return {
                "page_num": page_num - 1,
                "tables": [],
                "table_count": 0,
                "error": str(e),
            }

    def extract_tables_from_all_pages(self, flavor: str = "lattice") -> List[Dict[str, Any]]:
        """
        Extract tables from all pages in the document.
        Tries both lattice and stream flavors to get best results.

        Args:
            flavor: 'lattice' for tables with lines, 'stream' for tables without lines, 'auto' to try both

        Returns:
            List of dictionaries containing tables from each page
        """
        all_tables = []
        
        try:
            if flavor == "auto":
                # Try both flavors and combine results
                logger.info("Trying lattice flavor first...")
                try:
                    lattice_tables = camelot.read_pdf(str(self.pdf_path), pages="all", flavor="lattice")
                    for idx, table in enumerate(lattice_tables):
                        if table.accuracy > 50:  # Only include if accuracy is decent
                            all_tables.append({
                                "table_num": len(all_tables),
                                "page_num": table.page - 1,
                                "dataframe": table.df,
                                "shape": table.df.shape,
                                "accuracy": table.accuracy,
                                "whitespace": table.whitespace,
                                "extraction_method": "lattice"
                            })
                except Exception as e:
                    logger.warning(f"Lattice extraction failed: {e}")
                
                logger.info("Trying stream flavor...")
                try:
                    stream_tables = camelot.read_pdf(str(self.pdf_path), pages="all", flavor="stream")
                    for idx, table in enumerate(stream_tables):
                        if table.accuracy > 50:  # Only include if accuracy is decent
                            all_tables.append({
                                "table_num": len(all_tables),
                                "page_num": table.page - 1,
                                "dataframe": table.df,
                                "shape": table.df.shape,
                                "accuracy": table.accuracy,
                                "whitespace": table.whitespace,
                                "extraction_method": "stream"
                            })
                except Exception as e:
                    logger.warning(f"Stream extraction failed: {e}")
            else:
                # Use specified flavor
                tables = camelot.read_pdf(str(self.pdf_path), pages="all", flavor=flavor)
                for idx, table in enumerate(tables):
                    all_tables.append({
                        "table_num": idx,
                        "page_num": table.page - 1,
                        "dataframe": table.df,
                        "shape": table.df.shape,
                        "accuracy": table.accuracy,
                        "whitespace": table.whitespace,
                        "extraction_method": flavor
                    })

            logger.info(f"Extracted {len(all_tables)} tables from entire document")
            return all_tables

        except Exception as e:
            logger.error(f"Error extracting tables from document: {e}")
            return []

    def table_to_text(self, df: pd.DataFrame) -> str:
        """
        Convert a table DataFrame to text format.

        Args:
            df: Pandas DataFrame

        Returns:
            String representation of the table
        """
        return df.to_string(index=False)

    def table_to_markdown(self, df: pd.DataFrame) -> str:
        """
        Convert a table DataFrame to markdown format.

        Args:
            df: Pandas DataFrame

        Returns:
            Markdown string representation of the table
        """
        return df.to_markdown(index=False)


if __name__ == "__main__":
    # Example usage
    pdf_path = "data/raw/qatar_test_doc.pdf"

    extractor = TableExtractor(pdf_path)

    # Extract tables from all pages
    all_tables = extractor.extract_tables_from_all_pages()
    print(f"Total tables found: {len(all_tables)}")

    # Display first table if any
    if all_tables:
        first_table = all_tables[0]
        print(f"\nFirst table (Page {first_table['page_num']}):")
        print(f"Shape: {first_table['shape']}")
        print(f"Accuracy: {first_table['accuracy']:.2f}")
        print("\nTable preview:")
        print(first_table["dataframe"].head())

        # Convert to text
        text_repr = extractor.table_to_text(first_table["dataframe"])
        print(f"\nText representation:\n{text_repr[:200]}...")
