"""
OCR Processor Module
Performs OCR on images using Tesseract via pytesseract.
"""

import pytesseract
from PIL import Image
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    Performs OCR on images to extract text.
    """

    def __init__(self, lang: str = "eng", config: str = ""):
        """
        Initialize the OCR processor.

        Args:
            lang: Language code for OCR (default: 'eng' for English)
            config: Additional Tesseract configuration string
        """
        self.lang = lang
        self.config = config

    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Perform OCR on a single PIL Image.

        Args:
            image: PIL Image object

        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            # Extract text
            text = pytesseract.image_to_string(
                image, lang=self.lang, config=self.config
            )

            # Get detailed data (includes confidence scores)
            data = pytesseract.image_to_data(
                image, lang=self.lang, config=self.config, output_type=pytesseract.Output.DICT
            )

            # Calculate average confidence (filter out -1 values which indicate no text)
            confidences = [
                int(conf) for conf in data["conf"] if int(conf) != -1
            ]
            avg_confidence = (
                sum(confidences) / len(confidences) if confidences else 0
            )

            result = {
                "text": text.strip(),
                "char_count": len(text.strip()),
                "word_count": len(text.split()),
                "avg_confidence": avg_confidence,
                "has_text": bool(text.strip()),
            }

            logger.info(
                f"OCR extracted {result['word_count']} words "
                f"(confidence: {avg_confidence:.2f}%)"
            )
            return result

        except Exception as e:
            logger.error(f"Error during OCR processing: {e}")
            return {
                "text": "",
                "char_count": 0,
                "word_count": 0,
                "avg_confidence": 0,
                "has_text": False,
                "error": str(e),
            }

    def process_image_from_path(self, image_path: str) -> Dict[str, Any]:
        """
        Perform OCR on an image file.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            image = Image.open(image_path)
            result = self.process_image(image)
            result["image_path"] = image_path
            return result

        except Exception as e:
            logger.error(f"Error loading image from {image_path}: {e}")
            return {
                "image_path": image_path,
                "text": "",
                "char_count": 0,
                "word_count": 0,
                "avg_confidence": 0,
                "has_text": False,
                "error": str(e),
            }

    def process_multiple_images(
        self, image_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Perform OCR on multiple images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of dictionaries with OCR results
        """
        results = []

        for image_path in image_paths:
            result = self.process_image_from_path(image_path)
            results.append(result)

        total_words = sum(r["word_count"] for r in results)
        logger.info(
            f"Processed {len(image_paths)} images, extracted {total_words} words total"
        )

        return results

    def process_images_from_directory(
        self, directory: str, extensions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform OCR on all images in a directory.

        Args:
            directory: Path to directory containing images
            extensions: List of file extensions to process (default: common image formats)

        Returns:
            List of dictionaries with OCR results
        """
        if extensions is None:
            extensions = [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]

        dir_path = Path(directory)
        image_paths = []

        for ext in extensions:
            image_paths.extend([str(p) for p in dir_path.glob(f"*{ext}")])

        logger.info(f"Found {len(image_paths)} images in {directory}")

        return self.process_multiple_images(image_paths)


if __name__ == "__main__":
    # Example usage
    ocr = OCRProcessor(lang="eng")

    # Process images from the intermediate directory
    image_dir = "data/intermediate/images"

    if Path(image_dir).exists():
        results = ocr.process_images_from_directory(image_dir)

        print(f"Processed {len(results)} images\n")

        for result in results:
            if result["has_text"]:
                print(f"Image: {Path(result['image_path']).name}")
                print(f"Words: {result['word_count']}")
                print(f"Confidence: {result['avg_confidence']:.2f}%")
                print(f"Text preview: {result['text'][:100]}...")
                print("-" * 50)
    else:
        print(f"Directory {image_dir} not found. Extract images first.")
