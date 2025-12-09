"""
CLI App
Command-line interface for the Multi-Modal RAG QA System.
"""

import sys
from pathlib import Path
import argparse
import os

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent / "rag"))
sys.path.append(str(Path(__file__).parent.parent / "embeddings"))
sys.path.append(str(Path(__file__).parent.parent / "vectorstore"))

from pipeline import RAGPipeline
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

console = Console()


def display_result(result: dict, show_context: bool = False, show_chunks: bool = False):
    """Display query result in a formatted way."""
    # Display question
    console.print(Panel(result["question"], title="❓ Question", border_style="blue"))

    # Display answer
    console.print(Panel(Markdown(result["answer"]), title="💡 Answer", border_style="green"))

    # Display context summary
    summary = result.get("context_summary", {})
    if summary:
        table = Table(title="📊 Context Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Total Chunks", str(summary.get("total_chunks", 0)))
        table.add_row("Pages Referenced", str(summary.get("pages_referenced", [])))
        table.add_row("Content Types", str(summary.get("content_types", {})))

        console.print(table)

    # Display token usage
    usage = result.get("usage", {})
    if usage:
        console.print(
            f"\n[yellow]Tokens Used:[/yellow] {usage.get('total_tokens', 'N/A')} "
            f"(Prompt: {usage.get('prompt_tokens', 'N/A')}, "
            f"Completion: {usage.get('completion_tokens', 'N/A')})"
        )

    # Display context if requested
    if show_context and "context" in result:
        console.print(Panel(result["context"], title="📄 Retrieved Context", border_style="yellow"))

    # Display chunks if requested
    if show_chunks and "retrieved_chunks" in result:
        console.print("\n[bold cyan]🔍 Retrieved Chunks:[/bold cyan]")
        for chunk in result["retrieved_chunks"]:
            meta = chunk["metadata"]
            console.print(
                f"\n[bold]Rank {chunk['rank']}[/bold] "
                f"(Similarity: {chunk['similarity']:.3f})"
            )
            console.print(
                f"Type: {meta.get('content_type', 'unknown')}, "
                f"Page: {meta.get('page_num', '?')}"
            )
            text = meta.get("text", meta.get("table_text", meta.get("ocr_text", "N/A")))
            console.print(f"{text[:200]}...\n")


def interactive_mode(pipeline: RAGPipeline, args):
    """Run in interactive mode."""
    console.print(
        Panel(
            "[bold green]Multi-Modal RAG QA System[/bold green]\n"
            "Type your questions or 'quit' to exit.",
            border_style="green",
        )
    )

    while True:
        try:
            # Get question from user
            question = console.input("\n[bold cyan]Question:[/bold cyan] ")

            if question.lower() in ["quit", "exit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            if not question.strip():
                continue

            # Process query
            console.print("\n[dim]Processing...[/dim]")
            result = pipeline.query(
                question=question,
                top_k=args.top_k,
                content_type_filter=args.content_type,
                return_context=args.show_context or args.show_chunks,
            )

            # Display result
            console.print()
            display_result(result, args.show_context, args.show_chunks)

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def single_query_mode(pipeline: RAGPipeline, args):
    """Process a single query."""
    result = pipeline.query(
        question=args.question,
        top_k=args.top_k,
        content_type_filter=args.content_type,
        return_context=args.show_context or args.show_chunks,
    )

    display_result(result, args.show_context, args.show_chunks)


def show_stats(pipeline: RAGPipeline):
    """Display pipeline statistics."""
    stats = pipeline.get_pipeline_stats()

    table = Table(title="📊 Pipeline Statistics", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Details", style="magenta")

    table.add_row("Text Embedding Model", stats.get("text_embedding_model", "N/A"))
    table.add_row("Embedding Dimension", str(stats.get("text_embedding_dim", "N/A")))
    table.add_row("LLM Model", stats.get("llm_model", "N/A"))

    index_stats = stats.get("index_stats", {})
    table.add_row("Total Vectors", str(index_stats.get("total_vectors", "N/A")))
    table.add_row("Content Types", str(index_stats.get("content_types", {})))

    console.print(table)


def main():
    """Main CLI application."""
    parser = argparse.ArgumentParser(
        description="Multi-Modal RAG QA System - Command Line Interface"
    )

    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Question to ask (if not provided, runs in interactive mode)",
    )

    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )

    parser.add_argument(
        "-t", "--content-type",
        type=str,
        choices=["text", "image", "table"],
        help="Filter by content type",
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Show retrieved context",
    )

    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Show individual retrieved chunks",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show pipeline statistics and exit",
    )

    parser.add_argument(
        "--index-path",
        type=str,
        default="data/index/faiss_index.bin",
        help="Path to FAISS index",
    )

    parser.add_argument(
        "--metadata-path",
        type=str,
        default="data/index/metadata.pkl",
        help="Path to metadata file",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.1-70b-versatile",
        help="Groq LLM model to use",
    )

    args = parser.parse_args()

    # Check if index exists
    if not Path(args.index_path).exists() or not Path(args.metadata_path).exists():
        console.print(
            "[red]Error: Index files not found![/red]\n"
            "Please build the index first by running the ingestion pipeline."
        )
        sys.exit(1)

    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        console.print(
            "[red]Error: GROQ_API_KEY environment variable not set![/red]\n"
            "Set it with: export GROQ_API_KEY='your-api-key-here'"
        )
        sys.exit(1)

    try:
        # Initialize pipeline
        console.print("[dim]Initializing pipeline...[/dim]")
        pipeline = RAGPipeline(
            index_path=args.index_path,
            metadata_path=args.metadata_path,
            llm_model=args.model,
        )
        console.print("[green]✓ Pipeline initialized[/green]\n")

        # Show stats if requested
        if args.stats:
            show_stats(pipeline)
            sys.exit(0)

        # Run in appropriate mode
        if args.question:
            single_query_mode(pipeline, args)
        else:
            interactive_mode(pipeline, args)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
