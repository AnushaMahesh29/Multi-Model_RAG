"""
Context Assembler Module
Assembles retrieved content into a coherent context for the LLM.
"""

from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextAssembler:
    """
    Assembles retrieved chunks into a formatted context for LLM generation.
    """

    def __init__(
        self,
        max_context_length: int = 4000,
        include_metadata: bool = True,
        separator: str = "\n\n---\n\n",
    ):
        """
        Initialize the context assembler.

        Args:
            max_context_length: Maximum character length for context
            include_metadata: Whether to include metadata in context
            separator: Separator between chunks
        """
        self.max_context_length = max_context_length
        self.include_metadata = include_metadata
        self.separator = separator

    def assemble(self, retrieved_results: List[Dict[str, Any]]) -> str:
        """
        Assemble retrieved results into a context string.

        Args:
            retrieved_results: List of retrieval results with metadata

        Returns:
            Formatted context string
        """
        if not retrieved_results:
            return ""

        context_parts = []
        current_length = 0

        for result in retrieved_results:
            # Format this chunk
            chunk_text = self._format_chunk(result)

            # Skip empty chunks
            if not chunk_text or not chunk_text.strip():
                continue

            # Check if adding this chunk exceeds max length
            chunk_length = len(chunk_text) + len(self.separator)
            if current_length + chunk_length > self.max_context_length:
                logger.info(
                    f"Reached max context length. Including {len(context_parts)} chunks."
                )
                break

            context_parts.append(chunk_text)
            current_length += chunk_length

        # Join all parts
        context = self.separator.join(context_parts)

        logger.info(
            f"Assembled context with {len(context_parts)} chunks, "
            f"{len(context)} characters"
        )

        return context

    def _format_chunk(self, result: Dict[str, Any]) -> str:
        """
        Format a single retrieved chunk.

        Args:
            result: Retrieval result dictionary

        Returns:
            Formatted chunk string
        """
        metadata = result.get("metadata", {})
        content_type = metadata.get("content_type", "unknown")

        # Format based on content type
        if content_type == "text":
            return self._format_text_chunk(result)
        elif content_type == "table":
            return self._format_table_chunk(result)
        elif content_type == "image":
            return self._format_image_chunk(result)
        else:
            return self._format_generic_chunk(result)

    def _format_text_chunk(self, result: Dict[str, Any]) -> str:
        """Format a text chunk."""
        metadata = result["metadata"]
        text = metadata.get("text", "")

        if self.include_metadata:
            page_num = metadata.get("page_num", "?")
            chunk_idx = metadata.get("chunk_index", "?")
            header = f"[Text from Page {page_num}, Chunk {chunk_idx}]"
            return f"{header}\n{text}"
        else:
            return text

    def _format_table_chunk(self, result: Dict[str, Any]) -> str:
        """Format a table chunk."""
        metadata = result["metadata"]
        table_text = metadata.get("table_text", "").strip()

        if not table_text:
            # If table_text is missing, log warning and skip
            logger.warning(f"Table chunk missing table_text: {metadata}")
            return ""

        if self.include_metadata:
            page_num = metadata.get("page_num", "?")
            table_num = metadata.get("table_num", "?")
            header = f"[Table from Page {page_num}, Table {table_num}]"
            return f"{header}\n\n{table_text}"
        else:
            return table_text

    def _format_image_chunk(self, result: Dict[str, Any]) -> str:
        """Format an image chunk (using OCR text if available)."""
        metadata = result["metadata"]

        # Try to get OCR text if available
        ocr_text = metadata.get("ocr_text", "").strip()

        if self.include_metadata:
            page_num = metadata.get("page_num", "?")
            img_num = metadata.get("image_num", "?")
            header = f"[Image from Page {page_num}, Image {img_num}]"

            if ocr_text:
                return f"{header}\nOCR Text:\n{ocr_text}"
            else:
                # Don't include images without OCR text as they provide no textual context
                return ""
        else:
            return ocr_text if ocr_text else ""

    def _format_generic_chunk(self, result: Dict[str, Any]) -> str:
        """Format a generic chunk."""
        metadata = result["metadata"]
        text = metadata.get("text", str(metadata))

        if self.include_metadata:
            return f"[Content]\n{text}"
        else:
            return text

    def assemble_with_query(
        self, query: str, retrieved_results: List[Dict[str, Any]]
    ) -> str:
        """
        Assemble context with the query included.

        Args:
            query: User query
            retrieved_results: List of retrieval results

        Returns:
            Formatted context with query
        """
        context = self.assemble(retrieved_results)

        formatted = f"Query: {query}\n\n"
        formatted += "Relevant Context:\n"
        formatted += context

        return formatted

    def get_context_summary(self, retrieved_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get a summary of the assembled context.

        Args:
            retrieved_results: List of retrieval results

        Returns:
            Dictionary with context statistics
        """
        content_types = {}
        pages = set()

        for result in retrieved_results:
            metadata = result.get("metadata", {})

            # Count content types
            ctype = metadata.get("content_type", "unknown")
            content_types[ctype] = content_types.get(ctype, 0) + 1

            # Collect pages
            page_num = metadata.get("page_num")
            if page_num is not None:
                pages.add(page_num)

        return {
            "total_chunks": len(retrieved_results),
            "content_types": content_types,
            "pages_referenced": sorted(list(pages)),
            "num_pages": len(pages),
        }


if __name__ == "__main__":
    # Example usage
    assembler = ContextAssembler(max_context_length=1000, include_metadata=True)

    # Sample retrieved results
    sample_results = [
        {
            "rank": 1,
            "similarity": 0.95,
            "metadata": {
                "text": "This is a sample text chunk about machine learning.",
                "page_num": 0,
                "chunk_index": 0,
                "content_type": "text",
            },
        },
        {
            "rank": 2,
            "similarity": 0.87,
            "metadata": {
                "table_text": "| Model | Accuracy |\n|-------|----------|\n| A | 95% |\n| B | 92% |",
                "page_num": 1,
                "table_num": 0,
                "content_type": "table",
            },
        },
        {
            "rank": 3,
            "similarity": 0.82,
            "metadata": {
                "ocr_text": "Figure 1: Neural network architecture",
                "page_num": 2,
                "image_num": 0,
                "content_type": "image",
            },
        },
    ]

    # Assemble context
    context = assembler.assemble(sample_results)
    print("Assembled Context:")
    print(context)

    # Get summary
    summary = assembler.get_context_summary(sample_results)
    print("\n\nContext Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Assemble with query
    query = "What is the accuracy of model A?"
    context_with_query = assembler.assemble_with_query(query, sample_results)
    print("\n\nContext with Query:")
    print(context_with_query[:200] + "...")
