"""
Text Chunker Module
Splits text into smaller chunks for embedding and retrieval.
"""

from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextChunker:
    """
    Chunks text into smaller pieces suitable for embedding.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = "\n",
    ):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
            separator: Separator to use for splitting (default: newline)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with overlap.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of dictionaries containing chunks and metadata
        """
        if not text or not text.strip():
            return []

        if metadata is None:
            metadata = {}

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate end position
            end = start + self.chunk_size

            # If this is not the last chunk, try to break at a separator
            if end < text_length:
                # Look for separator within the chunk
                separator_pos = text.rfind(self.separator, start, end)
                if separator_pos != -1 and separator_pos > start:
                    end = separator_pos + len(self.separator)

            # Extract chunk
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_data = {
                    "text": chunk_text,
                    "chunk_index": len(chunks),
                    "start_char": start,
                    "end_char": end,
                    "char_count": len(chunk_text),
                    **metadata,
                }
                chunks.append(chunk_data)

            # Move start position with overlap
            start = end - self.chunk_overlap if end < text_length else text_length

        logger.info(f"Created {len(chunks)} chunks from text of length {text_length}")
        return chunks

    def chunk_by_sentences(
        self, text: str, max_sentences: int = 5, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks by sentences.

        Args:
            text: Input text to chunk
            max_sentences: Maximum number of sentences per chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of dictionaries containing chunks and metadata
        """
        if not text or not text.strip():
            return []

        if metadata is None:
            metadata = {}

        # Simple sentence splitting (can be improved with NLTK)
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence exceeds chunk_size, save current chunk
            if current_length + len(sentence) > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_data = {
                    "text": chunk_text,
                    "chunk_index": len(chunks),
                    "sentence_count": len(current_chunk),
                    "char_count": len(chunk_text),
                    **metadata,
                }
                chunks.append(chunk_data)
                current_chunk = []
                current_length = 0

            current_chunk.append(sentence)
            current_length += len(sentence)

            # If we've reached max_sentences, save chunk
            if len(current_chunk) >= max_sentences:
                chunk_text = " ".join(current_chunk)
                chunk_data = {
                    "text": chunk_text,
                    "chunk_index": len(chunks),
                    "sentence_count": len(current_chunk),
                    "char_count": len(chunk_text),
                    **metadata,
                }
                chunks.append(chunk_data)
                current_chunk = []
                current_length = 0

        # Add remaining sentences
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_data = {
                "text": chunk_text,
                "chunk_index": len(chunks),
                "sentence_count": len(current_chunk),
                "char_count": len(chunk_text),
                **metadata,
            }
            chunks.append(chunk_data)

        logger.info(f"Created {len(chunks)} sentence-based chunks")
        return chunks

    def chunk_by_paragraphs(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks by paragraphs.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of dictionaries containing chunks and metadata
        """
        if not text or not text.strip():
            return []

        if metadata is None:
            metadata = {}

        # Split by double newlines (paragraphs)
        paragraphs = text.split("\n\n")

        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds chunk_size, save current chunk
            if current_length + len(para) > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunk_data = {
                    "text": chunk_text,
                    "chunk_index": len(chunks),
                    "paragraph_count": len(current_chunk),
                    "char_count": len(chunk_text),
                    **metadata,
                }
                chunks.append(chunk_data)
                current_chunk = []
                current_length = 0

            current_chunk.append(para)
            current_length += len(para)

        # Add remaining paragraphs
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk_data = {
                "text": chunk_text,
                "chunk_index": len(chunks),
                "paragraph_count": len(current_chunk),
                "char_count": len(chunk_text),
                **metadata,
            }
            chunks.append(chunk_data)

        logger.info(f"Created {len(chunks)} paragraph-based chunks")
        return chunks


if __name__ == "__main__":
    # Example usage
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)

    sample_text = """
    This is the first paragraph. It contains multiple sentences. 
    Each sentence adds information.
    
    This is the second paragraph. It also has several sentences.
    We want to chunk this text intelligently.
    
    This is the third paragraph. Chunking helps with embedding.
    Smaller chunks are better for retrieval.
    """

    # Character-based chunking
    chunks = chunker.chunk_text(
        sample_text, metadata={"page_num": 0, "source": "test"}
    )
    print(f"Character-based chunks: {len(chunks)}")
    for chunk in chunks:
        print(f"\nChunk {chunk['chunk_index']}:")
        print(f"  Text: {chunk['text'][:50]}...")
        print(f"  Chars: {chunk['char_count']}")

    # Sentence-based chunking
    sentence_chunks = chunker.chunk_by_sentences(sample_text, max_sentences=3)
    print(f"\n\nSentence-based chunks: {len(sentence_chunks)}")

    # Paragraph-based chunking
    para_chunks = chunker.chunk_by_paragraphs(sample_text)
    print(f"Paragraph-based chunks: {len(para_chunks)}")
