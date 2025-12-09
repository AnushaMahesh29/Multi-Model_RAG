"""
Text Cleaner Module
Cleans and normalizes extracted text from PDFs, OCR, and tables.
"""

import re
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Cleans and normalizes text content.
    """

    def __init__(
        self,
        remove_extra_whitespace: bool = True,
        remove_special_chars: bool = False,
        lowercase: bool = False,
        remove_urls: bool = True,
        remove_emails: bool = False,
    ):
        """
        Initialize the text cleaner.

        Args:
            remove_extra_whitespace: Remove extra spaces, tabs, newlines
            remove_special_chars: Remove special characters (keep alphanumeric and basic punctuation)
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses from text
        """
        self.remove_extra_whitespace = remove_extra_whitespace
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails

    def clean(self, text: str) -> str:
        """
        Clean a single text string.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""

        cleaned = text

        # Remove URLs
        if self.remove_urls:
            cleaned = re.sub(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                "",
                cleaned,
            )

        # Remove emails
        if self.remove_emails:
            cleaned = re.sub(r"\S+@\S+", "", cleaned)

        # Remove special characters (keep alphanumeric, spaces, and basic punctuation)
        if self.remove_special_chars:
            cleaned = re.sub(r"[^a-zA-Z0-9\s.,!?;:()\-]", "", cleaned)

        # Remove extra whitespace
        if self.remove_extra_whitespace:
            # Replace multiple spaces with single space
            cleaned = re.sub(r" +", " ", cleaned)
            # Replace multiple newlines with single newline
            cleaned = re.sub(r"\n+", "\n", cleaned)
            # Remove leading/trailing whitespace
            cleaned = cleaned.strip()

        # Convert to lowercase
        if self.lowercase:
            cleaned = cleaned.lower()

        return cleaned

    def clean_multiple(self, texts: List[str]) -> List[str]:
        """
        Clean multiple text strings.

        Args:
            texts: List of input texts

        Returns:
            List of cleaned texts
        """
        return [self.clean(text) for text in texts]

    def remove_page_numbers(self, text: str) -> str:
        """
        Remove common page number patterns.

        Args:
            text: Input text

        Returns:
            Text with page numbers removed
        """
        # Remove standalone numbers at start or end of lines
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        # Remove "Page X" patterns
        text = re.sub(r"Page\s+\d+", "", text, flags=re.IGNORECASE)
        return text

    def remove_headers_footers(
        self, text: str, header_pattern: Optional[str] = None, footer_pattern: Optional[str] = None
    ) -> str:
        """
        Remove headers and footers based on patterns.

        Args:
            text: Input text
            header_pattern: Regex pattern for headers
            footer_pattern: Regex pattern for footers

        Returns:
            Text with headers/footers removed
        """
        if header_pattern:
            text = re.sub(header_pattern, "", text, flags=re.IGNORECASE)

        if footer_pattern:
            text = re.sub(footer_pattern, "", text, flags=re.IGNORECASE)

        return text

    def normalize_unicode(self, text: str) -> str:
        """
        Normalize unicode characters to ASCII equivalents.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Replace common unicode characters
        replacements = {
            "\u2018": "'",  # Left single quote
            "\u2019": "'",  # Right single quote
            "\u201c": '"',  # Left double quote
            "\u201d": '"',  # Right double quote
            "\u2013": "-",  # En dash
            "\u2014": "-",  # Em dash
            "\u2026": "...",  # Ellipsis
        }

        for unicode_char, ascii_char in replacements.items():
            text = text.replace(unicode_char, ascii_char)

        return text


if __name__ == "__main__":
    # Example usage
    cleaner = TextCleaner(
        remove_extra_whitespace=True,
        remove_urls=True,
        lowercase=False,
    )

    sample_text = """
    This is    a sample   text with   extra    spaces.
    
    
    Multiple newlines above.
    
    Visit https://example.com for more info.
    Contact us at test@example.com
    
    Page 1
    """

    cleaned = cleaner.clean(sample_text)
    print("Original text:")
    print(repr(sample_text))
    print("\nCleaned text:")
    print(repr(cleaned))

    # Remove page numbers
    cleaned = cleaner.remove_page_numbers(cleaned)
    print("\nAfter removing page numbers:")
    print(repr(cleaned))

    # Normalize unicode
    unicode_text = "It's a \"test\" — with unicode…"
    normalized = cleaner.normalize_unicode(unicode_text)
    print(f"\nUnicode text: {unicode_text}")
    print(f"Normalized: {normalized}")
