"""
Text Embedder Module
Generates embeddings for text using SentenceTransformers.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    Generates embeddings for text chunks using SentenceTransformers.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize the text embedder.

        Args:
            model_name: Name of the SentenceTransformer model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device

        logger.info(f"Loading text embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.

        Args:
            text: Input text

        Returns:
            Numpy array containing the embedding
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.embedding_dim)

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for encoding

        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        logger.info(f"Embedding {len(texts)} texts...")
        embeddings = self.model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True
        )

        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def embed_chunks(
        self, chunks: List[Dict[str, Any]], batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for text chunks (with metadata).

        Args:
            chunks: List of chunk dictionaries (must contain 'text' key)
            batch_size: Batch size for encoding

        Returns:
            List of chunk dictionaries with added 'embedding' key
        """
        if not chunks:
            return []

        # Extract texts
        texts = [chunk["text"] for chunk in chunks]

        # Generate embeddings
        embeddings = self.embed_texts(texts, batch_size=batch_size)

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
            chunk["embedding_model"] = self.model_name

        logger.info(f"Added embeddings to {len(chunks)} chunks")
        return chunks

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.

        Returns:
            Embedding dimension
        """
        return self.embedding_dim

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)


if __name__ == "__main__":
    # Example usage
    embedder = TextEmbedder(model_name="all-MiniLM-L6-v2", device="cpu")

    # Single text embedding
    text = "This is a sample text for embedding."
    embedding = embedder.embed_text(text)
    print(f"Single text embedding shape: {embedding.shape}")
    print(f"Embedding preview: {embedding[:5]}")

    # Multiple texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand text.",
    ]

    embeddings = embedder.embed_texts(texts)
    print(f"\nMultiple texts embeddings shape: {embeddings.shape}")

    # Compute similarity
    similarity = embedder.compute_similarity(embeddings[0], embeddings[1])
    print(f"\nSimilarity between text 0 and 1: {similarity:.4f}")

    # Embed chunks with metadata
    chunks = [
        {"text": texts[0], "page_num": 0, "chunk_index": 0},
        {"text": texts[1], "page_num": 0, "chunk_index": 1},
        {"text": texts[2], "page_num": 1, "chunk_index": 0},
    ]

    embedded_chunks = embedder.embed_chunks(chunks)
    print(f"\nEmbedded {len(embedded_chunks)} chunks")
    print(f"First chunk keys: {embedded_chunks[0].keys()}")
