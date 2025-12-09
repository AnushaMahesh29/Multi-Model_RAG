"""
Retriever Module
Performs similarity search to retrieve relevant content from FAISS index.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging
from faiss_loader import FAISSLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves relevant content from FAISS index based on query embeddings.
    """

    def __init__(self, faiss_loader: FAISSLoader):
        """
        Initialize the retriever.

        Args:
            faiss_loader: FAISSLoader instance with loaded index and metadata
        """
        self.loader = faiss_loader
        self.index = faiss_loader.get_index()
        self.metadata = faiss_loader.get_metadata()

        logger.info(f"Retriever initialized with {self.index.ntotal} vectors")

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the index.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return

        Returns:
            List of dictionaries with results and metadata
        """
        # Ensure query is 2D array and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)
        
        # Pad query embedding if needed to match index dimension
        index_dim = self.index.d
        query_dim = query_embedding.shape[1]
        
        if query_dim < index_dim:
            # Pad with zeros to match index dimension
            padding = np.zeros((query_embedding.shape[0], index_dim - query_dim), dtype=np.float32)
            query_embedding = np.concatenate([query_embedding, padding], axis=1)
            logger.info(f"Padded query embedding from {query_dim} to {index_dim} dimensions")
        elif query_dim > index_dim:
            # Truncate if query is larger (shouldn't happen, but handle it)
            query_embedding = query_embedding[:, :index_dim]
            logger.warning(f"Truncated query embedding from {query_dim} to {index_dim} dimensions")

        # Perform search
        distances, indices = self.index.search(query_embedding, top_k)

        # Prepare results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for empty results
                continue

            result = {
                "rank": i + 1,
                "index": int(idx),
                "distance": float(dist),
                "similarity": self._distance_to_similarity(float(dist)),
                "metadata": self.loader.get_metadata_by_id(int(idx)),
            }
            results.append(result)

        logger.info(f"Retrieved {len(results)} results")
        return results

    def search_batch(
        self, query_embeddings: np.ndarray, top_k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for multiple queries at once.

        Args:
            query_embeddings: Array of query embeddings (n_queries, embedding_dim)
            top_k: Number of top results to return per query

        Returns:
            List of result lists, one per query
        """
        # Ensure embeddings are float32
        query_embeddings = query_embeddings.astype(np.float32)

        # Perform batch search
        distances, indices = self.index.search(query_embeddings, top_k)

        # Prepare results for each query
        all_results = []
        for query_idx in range(len(query_embeddings)):
            query_results = []
            for i, (dist, idx) in enumerate(
                zip(distances[query_idx], indices[query_idx])
            ):
                if idx == -1:
                    continue

                result = {
                    "rank": i + 1,
                    "index": int(idx),
                    "distance": float(dist),
                    "similarity": self._distance_to_similarity(float(dist)),
                    "metadata": self.loader.get_metadata_by_id(int(idx)),
                }
                query_results.append(result)

            all_results.append(query_results)

        logger.info(f"Retrieved results for {len(all_results)} queries")
        return all_results

    def search_with_filter(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_fn: Optional[callable] = None,
        max_candidates: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Search with metadata filtering.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return after filtering
            filter_fn: Function that takes metadata dict and returns True/False
            max_candidates: Number of candidates to retrieve before filtering

        Returns:
            List of filtered results
        """
        # Get more candidates than needed
        candidates = self.search(query_embedding, top_k=max_candidates)

        # Apply filter if provided
        if filter_fn:
            filtered = [c for c in candidates if filter_fn(c["metadata"])]
            logger.info(
                f"Filtered {len(candidates)} candidates to {len(filtered)} results"
            )
        else:
            filtered = candidates

        # Return top_k results
        return filtered[:top_k]

    def search_by_content_type(
        self, query_embedding: np.ndarray, content_type: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for specific content type (text, image, table).

        Args:
            query_embedding: Query embedding vector
            content_type: Type of content to retrieve ('text', 'image', 'table')
            top_k: Number of top results to return

        Returns:
            List of results filtered by content type
        """
        filter_fn = lambda meta: meta.get("content_type") == content_type
        return self.search_with_filter(query_embedding, top_k, filter_fn)

    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert L2 distance to similarity score (0-1 range).

        Args:
            distance: L2 distance

        Returns:
            Similarity score
        """
        # For L2 distance, convert to similarity using exponential decay
        # Smaller distance = higher similarity
        similarity = np.exp(-distance / 10.0)
        return float(similarity)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retriever.

        Returns:
            Dictionary with statistics
        """
        # Count content types
        content_types = {}
        for meta in self.metadata:
            ctype = meta.get("content_type", "unknown")
            content_types[ctype] = content_types.get(ctype, 0) + 1

        return {
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.index.d,
            "content_types": content_types,
        }


if __name__ == "__main__":
    # Example usage
    from pathlib import Path

    index_path = "data/index/faiss_index.bin"
    metadata_path = "data/index/metadata.pkl"

    if Path(index_path).exists() and Path(metadata_path).exists():
        # Load index
        with FAISSLoader(index_path, metadata_path) as loader:
            # Create retriever
            retriever = Retriever(loader)

            # Get stats
            stats = retriever.get_stats()
            print("Retriever stats:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

            # Create sample query embedding
            embedding_dim = retriever.index.d
            query_embedding = np.random.randn(embedding_dim).astype(np.float32)

            # Search
            results = retriever.search(query_embedding, top_k=3)
            print(f"\nTop 3 results:")
            for result in results:
                print(f"  Rank {result['rank']}: "
                      f"similarity={result['similarity']:.4f}, "
                      f"type={result['metadata'].get('content_type', 'unknown')}")

            # Search by content type
            text_results = retriever.search_by_content_type(
                query_embedding, "text", top_k=2
            )
            print(f"\nTop 2 text results: {len(text_results)} found")
    else:
        print("Index files not found. Build an index first.")
