"""
FAISS Loader Module
Loads FAISS vector index and metadata from disk.
"""

import faiss
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSLoader:
    """
    Loads FAISS vector index and metadata from disk.
    """

    def __init__(self, index_path: str, metadata_path: str):
        """
        Initialize the FAISS loader.

        Args:
            index_path: Path to the FAISS index file
            metadata_path: Path to the metadata file
        """
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []

        # Check if files exist
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    def load(self):
        """
        Load the FAISS index and metadata.
        """
        # Load FAISS index
        logger.info(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        logger.info(f"Loaded index with {self.index.ntotal} vectors")

        # Load metadata
        logger.info(f"Loading metadata from {self.metadata_path}")
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        logger.info(f"Loaded {len(self.metadata)} metadata entries")

        # Verify counts match
        if self.index.ntotal != len(self.metadata):
            logger.warning(
                f"Mismatch: index has {self.index.ntotal} vectors "
                f"but metadata has {len(self.metadata)} entries"
            )

    def get_index(self) -> faiss.Index:
        """
        Get the loaded FAISS index.

        Returns:
            FAISS index
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load() first.")
        return self.index

    def get_metadata(self) -> List[Dict[str, Any]]:
        """
        Get the loaded metadata.

        Returns:
            List of metadata dictionaries
        """
        if not self.metadata:
            raise ValueError("Metadata not loaded. Call load() first.")
        return self.metadata

    def get_metadata_by_id(self, idx: int) -> Dict[str, Any]:
        """
        Get metadata for a specific vector by its index.

        Args:
            idx: Index of the vector

        Returns:
            Metadata dictionary
        """
        if not self.metadata:
            raise ValueError("Metadata not loaded. Call load() first.")

        if idx < 0 or idx >= len(self.metadata):
            raise IndexError(f"Index {idx} out of range [0, {len(self.metadata)})")

        return self.metadata[idx]

    def get_metadata_by_ids(self, indices: List[int]) -> List[Dict[str, Any]]:
        """
        Get metadata for multiple vectors by their indices.

        Args:
            indices: List of vector indices

        Returns:
            List of metadata dictionaries
        """
        return [self.get_metadata_by_id(idx) for idx in indices]

    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded index.

        Returns:
            Dictionary with index information
        """
        if self.index is None:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.index.d,
            "metadata_count": len(self.metadata),
        }

    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # FAISS doesn't require explicit cleanup
        pass


if __name__ == "__main__":
    # Example usage
    index_path = "data/index/faiss_index.bin"
    metadata_path = "data/index/metadata.pkl"

    # Check if files exist
    if Path(index_path).exists() and Path(metadata_path).exists():
        # Load with context manager
        with FAISSLoader(index_path, metadata_path) as loader:
            # Get index info
            info = loader.get_index_info()
            print("Index info:")
            for key, value in info.items():
                print(f"  {key}: {value}")

            # Get sample metadata
            if loader.get_metadata():
                print("\nFirst metadata entry:")
                first_meta = loader.get_metadata_by_id(0)
                for key, value in first_meta.items():
                    if key != "embedding":  # Skip embedding for display
                        print(f"  {key}: {value}")

                # Get multiple metadata entries
                if len(loader.get_metadata()) >= 3:
                    print("\nFirst 3 metadata entries:")
                    meta_list = loader.get_metadata_by_ids([0, 1, 2])
                    for i, meta in enumerate(meta_list):
                        print(f"  Entry {i}: {meta.get('content_type', 'unknown')}")
    else:
        print(f"Index files not found at:")
        print(f"  {index_path}")
        print(f"  {metadata_path}")
        print("\nBuild an index first using faiss_builder.py")
