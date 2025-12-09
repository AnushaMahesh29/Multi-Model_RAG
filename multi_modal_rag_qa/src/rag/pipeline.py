"""
RAG Pipeline Module
Complete end-to-end RAG pipeline: query -> retrieve -> assemble -> generate.
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent / "embeddings"))
sys.path.append(str(Path(__file__).parent.parent / "vectorstore"))

from typing import Dict, Any, Optional
import logging
from text_embedder import TextEmbedder
from faiss_loader import FAISSLoader
from retriever import Retriever
from context_assembler import ContextAssembler
from generator import Generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline for multi-modal question answering.
    """

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        groq_api_key: Optional[str] = None,
        text_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-70b-versatile",
        device: str = "cpu",
    ):
        """
        Initialize the RAG pipeline.

        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata file
            groq_api_key: Groq API key
            text_model: SentenceTransformer model for query embedding
            llm_model: Groq LLM model name
            device: Device for embeddings ('cpu' or 'cuda')
        """
        logger.info("Initializing RAG Pipeline...")

        # Initialize text embedder for queries
        self.text_embedder = TextEmbedder(model_name=text_model, device=device)

        # Load FAISS index
        self.faiss_loader = FAISSLoader(index_path, metadata_path)
        self.faiss_loader.load()

        # Initialize retriever
        self.retriever = Retriever(self.faiss_loader)

        # Initialize context assembler
        self.context_assembler = ContextAssembler(
            max_context_length=4000, include_metadata=True
        )

        # Initialize generator
        self.generator = Generator(api_key=groq_api_key, model=llm_model)

        logger.info("RAG Pipeline initialized successfully")

    def query(
        self,
        question: str,
        top_k: int = 5,
        content_type_filter: Optional[str] = None,
        return_context: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline.

        Args:
            question: User question
            top_k: Number of chunks to retrieve
            content_type_filter: Filter by content type ('text', 'image', 'table')
            return_context: Whether to include retrieved context in response

        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing query: {question}")

        # Step 1: Embed the query
        query_embedding = self.text_embedder.embed_text(question)

        # Step 2: Retrieve relevant chunks
        if content_type_filter:
            retrieved = self.retriever.search_by_content_type(
                query_embedding, content_type_filter, top_k
            )
        else:
            retrieved = self.retriever.search(query_embedding, top_k)

        logger.info(f"Retrieved {len(retrieved)} chunks")

        # Step 3: Assemble context
        context = self.context_assembler.assemble(retrieved)
        context_summary = self.context_assembler.get_context_summary(retrieved)

        # Step 4: Generate answer
        generation_result = self.generator.generate(question, context)

        # Prepare response
        response = {
            "question": question,
            "answer": generation_result["answer"],
            "context_summary": context_summary,
            "model": generation_result["model"],
            "usage": generation_result.get("usage", {}),
        }

        # Include context if requested
        if return_context:
            response["context"] = context
            response["retrieved_chunks"] = retrieved

        return response

    def query_with_citations(
        self, question: str, top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Process a query and generate answer with citations.

        Args:
            question: User question
            top_k: Number of chunks to retrieve

        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing query with citations: {question}")

        # Embed query
        query_embedding = self.text_embedder.embed_text(question)

        # Retrieve chunks
        retrieved = self.retriever.search(query_embedding, top_k)

        # Assemble context
        context = self.context_assembler.assemble(retrieved)

        # Generate with citations
        generation_result = self.generator.generate_with_citations(question, context)

        return {
            "question": question,
            "answer": generation_result["answer"],
            "retrieved_chunks": retrieved,
            "model": generation_result["model"],
            "usage": generation_result.get("usage", {}),
        }

    def query_concise(self, question: str, top_k: int = 3) -> str:
        """
        Get a concise answer to a question.

        Args:
            question: User question
            top_k: Number of chunks to retrieve

        Returns:
            Concise answer string
        """
        # Embed query
        query_embedding = self.text_embedder.embed_text(question)

        # Retrieve chunks
        retrieved = self.retriever.search(query_embedding, top_k)

        # Assemble context
        context = self.context_assembler.assemble(retrieved)

        # Generate concise answer
        generation_result = self.generator.generate_concise(question, context)

        return generation_result["answer"]

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the pipeline.

        Returns:
            Dictionary with pipeline statistics
        """
        retriever_stats = self.retriever.get_stats()

        return {
            "text_embedding_model": self.text_embedder.model_name,
            "text_embedding_dim": self.text_embedder.embedding_dim,
            "llm_model": self.generator.model,
            "index_stats": retriever_stats,
        }


if __name__ == "__main__":
    # Example usage
    index_path = "data/index/faiss_index.bin"
    metadata_path = "data/index/metadata.pkl"

    if Path(index_path).exists() and Path(metadata_path).exists():
        try:
            # Initialize pipeline
            pipeline = RAGPipeline(
                index_path=index_path,
                metadata_path=metadata_path,
                llm_model="llama-3.1-8b-instant",
            )

            # Get pipeline stats
            stats = pipeline.get_pipeline_stats()
            print("Pipeline Stats:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

            # Example query
            question = "What is machine learning?"

            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print(f"{'='*60}")

            # Process query
            result = pipeline.query(question, top_k=3, return_context=False)

            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nContext Summary:")
            for key, value in result["context_summary"].items():
                print(f"  {key}: {value}")
            print(f"\nTokens used: {result['usage'].get('total_tokens', 'N/A')}")

        except ValueError as e:
            print(f"Error: {e}")
            print("\nMake sure to set GROQ_API_KEY environment variable")

    else:
        print("Index files not found. Build an index first.")
        print(f"Expected files:")
        print(f"  {index_path}")
        print(f"  {metadata_path}")
