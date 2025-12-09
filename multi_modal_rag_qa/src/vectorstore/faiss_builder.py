"""
FAISS Builder Module
Builds and saves FAISS vector index from embeddings.
"""

import faiss
import numpy as np
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSBuilder:
    """
    Builds FAISS vector index for efficient similarity search.
    """

    def __init__(self, embedding_dim: int, index_type: str = "flat"):
        """
        Initialize the FAISS builder.

        Args:
            embedding_dim: Dimension of the embeddings
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.metadata = []

        logger.info(f"Initialized FAISS builder with {index_type} index")

    def create_index(self, n_clusters: int = 100) -> faiss.Index:
        """
        Create a FAISS index.

        Args:
            n_clusters: Number of clusters for IVF index (only used if index_type='ivf')

        Returns:
            FAISS index
        """
        if self.index_type == "flat":
            # Flat L2 index (exact search)
            index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info("Created Flat L2 index (exact search)")

        elif self.index_type == "ivf":
            # IVF index (approximate search, faster for large datasets)
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, n_clusters)
            logger.info(f"Created IVF index with {n_clusters} clusters")

        elif self.index_type == "hnsw":
            # HNSW index (hierarchical navigable small world, very fast)
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            logger.info("Created HNSW index")

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        self.index = index
        return index

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Add embeddings to the index.

        Args:
            embeddings: Numpy array of shape (n_samples, embedding_dim)
            metadata: List of metadata dictionaries for each embedding
        """
        if self.index is None:
            self.create_index()

        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)

        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
            logger.info("Training complete")

        # Add embeddings to index
        self.index.add(embeddings)

        # Store metadata
        if metadata:
            self.metadata.extend(metadata)
        else:
            # Create empty metadata
            self.metadata.extend([{} for _ in range(len(embeddings))])

        logger.info(f"Added {len(embeddings)} embeddings to index")
        logger.info(f"Total embeddings in index: {self.index.ntotal}")

    def add_from_embedded_items(self, embedded_items: List[Dict[str, Any]]):
        """
        Add embeddings from a list of embedded items.
        Handles different embedding dimensions by padding to the maximum dimension.

        Args:
            embedded_items: List of dictionaries with 'embedding' key
        """
        if not embedded_items:
            logger.warning("No embedded items to add")
            return
        
        # Extract embeddings
        embeddings_list = [item["embedding"] for item in embedded_items]
        
        # Check if all embeddings have the same dimension
        dims = [len(emb) for emb in embeddings_list]
        max_dim = max(dims)
        min_dim = min(dims)
        
        if max_dim == min_dim:
            # All same dimension, can directly convert
            embeddings = np.array(embeddings_list)
        else:
            # Different dimensions - pad to max dimension with zeros
            logger.warning(
                f"Embeddings have different dimensions (min: {min_dim}, max: {max_dim}). "
                f"Padding to {max_dim} dimensions."
            )
            padded_embeddings = []
            for emb in embeddings_list:
                if len(emb) < max_dim:
                    # Pad with zeros
                    padded = np.pad(emb, (0, max_dim - len(emb)), mode='constant')
                    padded_embeddings.append(padded)
                else:
                    padded_embeddings.append(emb)
            
            embeddings = np.array(padded_embeddings)
        
        # Extract metadata (remove embedding and dataframe to save space)
        metadata = []
        for item in embedded_items:
            meta = {k: v for k, v in item.items() if k not in ["embedding", "dataframe"]}
            # Ensure table_text is preserved for tables
            if "content_type" in meta and meta["content_type"] == "table":
                if "table_text" not in meta and "dataframe" in item:
                    # If table_text wasn't added, create it from dataframe
                    import pandas as pd
                    df = item["dataframe"]
                    if isinstance(df, pd.DataFrame):
                        meta["table_text"] = df.to_markdown(index=False)
            metadata.append(meta)

        # Add to index
        self.add_embeddings(embeddings, metadata)

    def save(self, index_path: str, metadata_path: str):
        """
        Save the FAISS index and metadata to disk.

        Args:
            index_path: Path to save the FAISS index
            metadata_path: Path to save the metadata
        """
        if self.index is None:
            raise ValueError("No index to save. Add embeddings first.")

        # Create directories if needed
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, index_path)
        logger.info(f"Saved FAISS index to {index_path}")

        # Save metadata
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Saved metadata to {metadata_path}")

    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the index.

        Returns:
            Dictionary with index information
        """
        if self.index is None:
            return {"status": "not_created"}

        return {
            "status": "created",
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim,
            "total_vectors": self.index.ntotal,
            "is_trained": self.index.is_trained if hasattr(self.index, "is_trained") else True,
            "metadata_count": len(self.metadata),
        }


if __name__ == "__main__":
    # Example usage
    embedding_dim = 384  # Example dimension

    # Create builder
    builder = FAISSBuilder(embedding_dim=embedding_dim, index_type="flat")

    # Create sample embeddings
    n_samples = 100
    sample_embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)

    # Create sample metadata
    sample_metadata = [
        {
            "text": f"Sample text {i}",
            "page_num": i // 10,
            "chunk_index": i % 10,
            "content_type": "text",
        }
        for i in range(n_samples)
    ]

    # Add embeddings
    builder.add_embeddings(sample_embeddings, sample_metadata)

    # Get index info
    info = builder.get_index_info()
    print("Index info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Save index
    index_path = "data/index/faiss_index.bin"
    metadata_path = "data/index/metadata.pkl"

    builder.save(index_path, metadata_path)
    print(f"\nIndex saved to {index_path}")
    print(f"Metadata saved to {metadata_path}")
