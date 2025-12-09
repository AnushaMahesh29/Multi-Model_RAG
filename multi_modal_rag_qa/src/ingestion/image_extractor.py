"""
Image Extractor Module
Extracts images from PDF pages using PyMuPDF.
"""

import fitz  # PyMuPDF
from typing import List, Dict, Any
from pathlib import Path
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageExtractor:
    """
    Extracts images from PDF documents.
    """

    def __init__(self, doc: fitz.Document, output_dir: str = "data/intermediate/images"):
        """
        Initialize the image extractor.

        Args:
            doc: PyMuPDF Document object
            output_dir: Directory to save extracted images
        """
        self.doc = doc
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_images_from_page(
        self, page_num: int, save_images: bool = True
    ) -> Dict[str, Any]:
        """
        Extract images from a single page.

        Args:
            page_num: Page number (0-indexed)
            save_images: Whether to save images to disk

        Returns:
            Dictionary with page number and extracted images info
        """
        try:
            page = self.doc[page_num]
            image_list = page.get_images(full=True)

            extracted_images = []

            for img_index, img in enumerate(image_list):
                xref = img[0]  # Image reference number

                # Extract image bytes
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Create PIL Image
                pil_image = Image.open(io.BytesIO(image_bytes))

                image_info = {
                    "image_num": img_index,
                    "xref": xref,
                    "width": pil_image.width,
                    "height": pil_image.height,
                    "format": image_ext,
                    "mode": pil_image.mode,
                    "size_bytes": len(image_bytes),
                }

                # Save image if requested
                if save_images:
                    image_filename = (
                        f"page_{page_num}_img_{img_index}.{image_ext}"
                    )
                    image_path = self.output_dir / image_filename
                    pil_image.save(image_path)
                    image_info["saved_path"] = str(image_path)
                    logger.info(f"Saved image: {image_filename}")
                else:
                    # Store image in memory
                    image_info["pil_image"] = pil_image

                extracted_images.append(image_info)

            result = {
                "page_num": page_num,
                "images": extracted_images,
                "image_count": len(extracted_images),
            }

            logger.info(
                f"Extracted {len(extracted_images)} images from page {page_num}"
            )
            return result

        except Exception as e:
            logger.error(f"Error extracting images from page {page_num}: {e}")
            return {
                "page_num": page_num,
                "images": [],
                "image_count": 0,
                "error": str(e),
            }

    def extract_images_from_all_pages(
        self, save_images: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract images from all pages in the document.

        Args:
            save_images: Whether to save images to disk

        Returns:
            List of dictionaries containing images from each page
        """
        all_images = []

        for page_num in range(len(self.doc)):
            page_images = self.extract_images_from_page(page_num, save_images)
            if page_images["image_count"] > 0:
                all_images.append(page_images)

        total_images = sum(page["image_count"] for page in all_images)
        logger.info(f"Extracted {total_images} images from {len(all_images)} pages")

        return all_images

    def get_image_by_path(self, image_path: str) -> Image.Image:
        """
        Load an image from disk.

        Args:
            image_path: Path to the image file

        Returns:
            PIL Image object
        """
        return Image.open(image_path)


if __name__ == "__main__":
    # Example usage
    from pdf_loader import PDFLoader

    pdf_path = "data/raw/qatar_test_doc.pdf"

    with PDFLoader(pdf_path) as loader:
        extractor = ImageExtractor(loader.doc)

        # Extract images from all pages
        all_images = extractor.extract_images_from_all_pages(save_images=True)

        print(f"Total pages with images: {len(all_images)}")

        # Display info about extracted images
        for page_data in all_images:
            print(f"\nPage {page_data['page_num']}: {page_data['image_count']} images")
            for img in page_data["images"]:
                print(
                    f"  - Image {img['image_num']}: {img['width']}x{img['height']} "
                    f"({img['format']}, {img['size_bytes']} bytes)"
                )
