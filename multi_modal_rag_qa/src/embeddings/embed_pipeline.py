"""
Embedding Pipeline Module
Orchestrates the embedding process for all content types (text, images, tables).
"""

from typing import List, Dict, Any, Optional
import numpy as np
import logging
from text_embedder import TextEmbedder
from image_embedder import ImageEmbedder
from table_embedder import TableEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """
    Orchestrates embedding generation for multi-modal content.
    """

    def __init__(
        self,
        text_model: str = "all-MiniLM-L6-v2",
        image_model: str = "ViT-B/32",
        device: str = "cpu",
    ):
        """
        Initialize the embedding pipeline.

        Args:
            text_model: SentenceTransformer model name for text
            image_model: CLIP model name for images
            device: Device to run models on ('cpu' or 'cuda')
        """
        logger.info("Initializing Embedding Pipeline...")

        # Initialize embedders
        self.text_embedder = TextEmbedder(model_name=text_model, device=device)
        self.image_embedder = ImageEmbedder(model_name=image_model, device=device)
        self.table_embedder = TableEmbedder(text_embedder=self.text_embedder)

        logger.info("Embedding Pipeline initialized successfully")

    def embed_text_chunks(
        self, text_chunks: List[Dict[str, Any]], batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Embed text chunks.

        Args:
            text_chunks: List of text chunk dictionaries
            batch_size: Batch size for encoding

        Returns:
            List of chunks with embeddings added
        """
        logger.info(f"Embedding {len(text_chunks)} text chunks...")
        embedded_chunks = self.text_embedder.embed_chunks(text_chunks, batch_size)

        # Add content type
        for chunk in embedded_chunks:
            chunk["content_type"] = "text"

        return embedded_chunks

    def embed_images(
        self, image_data: List[Dict[str, Any]], batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Embed images.

        Args:
            image_data: List of image data dictionaries
            batch_size: Batch size for encoding

        Returns:
            List of image data with embeddings added
        """
        logger.info(f"Embedding {len(image_data)} images...")
        embedded_images = self.image_embedder.embed_image_data(image_data, batch_size)

        # Add content type
        for img in embedded_images:
            img["content_type"] = "image"

        return embedded_images

    def embed_tables(
        self, table_data: List[Dict[str, Any]], batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Embed tables.

        Args:
            table_data: List of table data dictionaries
            batch_size: Batch size for encoding

        Returns:
            List of table data with embeddings added
        """
        logger.info(f"Embedding {len(table_data)} tables...")
        embedded_tables = self.table_embedder.embed_table_data(table_data, batch_size)

        # Add content type
        for table in embedded_tables:
            table["content_type"] = "table"

        return embedded_tables

    def embed_all(
        self,
        text_chunks: Optional[List[Dict[str, Any]]] = None,
        image_data: Optional[List[Dict[str, Any]]] = None,
        table_data: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 32,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Embed all content types.

        Args:
            text_chunks: List of text chunk dictionaries
            image_data: List of image data dictionaries
            table_data: List of table data dictionaries
            batch_size: Batch size for encoding

        Returns:
            Dictionary with embedded content by type
        """
        results = {"text": [], "images": [], "tables": []}

        # Embed text chunks
        if text_chunks:
            results["text"] = self.embed_text_chunks(text_chunks, batch_size)

        # Embed images
        if image_data:
            results["images"] = self.embed_images(image_data, batch_size)

        # Embed tables
        if table_data:
            results["tables"] = self.embed_tables(table_data, batch_size)

        # Log summary
        total = len(results["text"]) + len(results["images"]) + len(results["tables"])
        logger.info(
            f"Embedded {total} items: "
            f"{len(results['text'])} text, "
            f"{len(results['images'])} images, "
            f"{len(results['tables'])} tables"
        )

        return results

    def combine_embeddings(
        self, embedded_data: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Combine all embedded content into a single list.

        Args:
            embedded_data: Dictionary with embedded content by type

        Returns:
            Flattened list of all embedded items
        """
        combined = []

        # Add text chunks
        combined.extend(embedded_data.get("text", []))

        # Add images
        combined.extend(embedded_data.get("images", []))

        # Add tables
        combined.extend(embedded_data.get("tables", []))

        logger.info(f"Combined {len(combined)} embedded items")
        return combined

    def get_embeddings_array(
        self, embedded_items: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Extract embeddings as a numpy array.
        Handles different embedding dimensions by padding to the maximum dimension.

        Args:
            embedded_items: List of items with 'embedding' key

        Returns:
            Numpy array of shape (n_items, max_embedding_dim)
        """
        if not embedded_items:
            return np.array([])
        
        embeddings = [item["embedding"] for item in embedded_items]
        
        # Check if all embeddings have the same dimension
        dims = [len(emb) for emb in embeddings]
        max_dim = max(dims)
        min_dim = min(dims)
        
        if max_dim == min_dim:
            # All same dimension, can directly convert
            return np.array(embeddings)
        else:
            # Different dimensions - pad to max dimension with zeros
            logger.warning(
                f"Embeddings have different dimensions (min: {min_dim}, max: {max_dim}). "
                f"Padding to {max_dim} dimensions."
            )
            padded_embeddings = []
            for emb in embeddings:
                if len(emb) < max_dim:
                    # Pad with zeros
                    padded = np.pad(emb, (0, max_dim - len(emb)), mode='constant')
                    padded_embeddings.append(padded)
                else:
                    padded_embeddings.append(emb)
            
            return np.array(padded_embeddings)

    def get_embedding_dimensions(self) -> Dict[str, int]:
        """
        Get embedding dimensions for each content type.

        Returns:
            Dictionary with embedding dimensions
        """
        return {
            "text": self.text_embedder.get_embedding_dimension(),
            "image": self.image_embedder.get_embedding_dimension(),
            "table": self.table_embedder.get_embedding_dimension(),
        }


if __name__ == "__main__":
    # Example usage
    pipeline = EmbeddingPipeline(device="cpu")

    # Check embedding dimensions
    dims = pipeline.get_embedding_dimensions()
    print("Embedding dimensions:")
    for content_type, dim in dims.items():
        print(f"  {content_type}: {dim}")

    # Sample data
    text_chunks = [
        {"text": "This is a sample text chunk.", "page_num": 0, "chunk_index": 0},
        {"text": "Another text chunk for testing.", "page_num": 0, "chunk_index": 1},
    ]

    # Embed text chunks
    embedded_text = pipeline.embed_text_chunks(text_chunks)
    print(f"\nEmbedded {len(embedded_text)} text chunks")

    # Embed all (with only text in this example)
    all_embedded = pipeline.embed_all(text_chunks=text_chunks)
    print(f"\nEmbedded all content:")
    print(f"  Text: {len(all_embedded['text'])}")
    print(f"  Images: {len(all_embedded['images'])}")
    print(f"  Tables: {len(all_embedded['tables'])}")

    # Combine embeddings
    combined = pipeline.combine_embeddings(all_embedded)
    print(f"\nCombined {len(combined)} items")

    # Get embeddings array
    embeddings_array = pipeline.get_embeddings_array(combined)
    print(f"Embeddings array shape: {embeddings_array.shape}")
