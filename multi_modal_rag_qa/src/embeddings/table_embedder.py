"""
Table Embedder Module
Generates embeddings for tables by converting them to text and using text embeddings.
"""

import pandas as pd
from typing import List, Dict, Any
import numpy as np
import logging
from text_embedder import TextEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TableEmbedder:
    """
    Generates embeddings for tables by converting them to text format.
    """

    def __init__(
        self,
        text_embedder: TextEmbedder = None,
        table_format: str = "markdown",
    ):
        """
        Initialize the table embedder.

        Args:
            text_embedder: TextEmbedder instance (creates new one if None)
            table_format: Format to convert tables ('markdown', 'string', 'csv')
        """
        self.table_format = table_format

        if text_embedder is None:
            logger.info("Creating new TextEmbedder for table embedding")
            self.text_embedder = TextEmbedder()
        else:
            self.text_embedder = text_embedder

        self.embedding_dim = self.text_embedder.get_embedding_dimension()

    def table_to_text(self, df: pd.DataFrame) -> str:
        """
        Convert a DataFrame to text format.

        Args:
            df: Pandas DataFrame

        Returns:
            String representation of the table
        """
        if df.empty:
            return ""

        if self.table_format == "markdown":
            # Convert to markdown format
            text = df.to_markdown(index=False)
        elif self.table_format == "csv":
            # Convert to CSV format
            text = df.to_csv(index=False)
        else:  # 'string' or default
            # Convert to string format
            text = df.to_string(index=False)

        return text

    def embed_table(self, df: pd.DataFrame, add_context: str = "") -> np.ndarray:
        """
        Generate embedding for a single table.

        Args:
            df: Pandas DataFrame
            add_context: Optional context to prepend (e.g., "Table showing sales data:")

        Returns:
            Numpy array containing the embedding
        """
        if df.empty:
            return np.zeros(self.embedding_dim)

        # Convert table to text
        table_text = self.table_to_text(df)

        # Add context if provided
        if add_context:
            table_text = f"{add_context}\n\n{table_text}"

        # Generate embedding
        embedding = self.text_embedder.embed_text(table_text)
        return embedding

    def embed_tables(
        self, tables: List[pd.DataFrame], batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for multiple tables.

        Args:
            tables: List of Pandas DataFrames
            batch_size: Batch size for encoding

        Returns:
            Numpy array of shape (n_tables, embedding_dim)
        """
        if not tables:
            return np.array([])

        # Convert all tables to text
        table_texts = [self.table_to_text(df) for df in tables]

        # Generate embeddings
        embeddings = self.text_embedder.embed_texts(table_texts, batch_size=batch_size)

        logger.info(f"Generated embeddings for {len(tables)} tables")
        return embeddings

    def embed_table_data(
        self, table_data: List[Dict[str, Any]], batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for table data with metadata.

        Args:
            table_data: List of dictionaries with 'dataframe' key
            batch_size: Batch size for encoding

        Returns:
            List of dictionaries with added 'embedding' and 'table_text' keys
        """
        if not table_data:
            return []

        # Extract DataFrames
        dataframes = [data["dataframe"] for data in table_data]

        # Convert to text
        table_texts = [self.table_to_text(df) for df in dataframes]

        # Generate embeddings
        embeddings = self.text_embedder.embed_texts(table_texts, batch_size=batch_size)

        # Add embeddings and text to data
        for data, embedding, text in zip(table_data, embeddings, table_texts):
            data["embedding"] = embedding
            data["table_text"] = text
            data["embedding_model"] = self.text_embedder.model_name

        logger.info(f"Added embeddings to {len(table_data)} tables")
        return table_data

    def embed_table_with_summary(
        self, df: pd.DataFrame, summary: str = ""
    ) -> np.ndarray:
        """
        Generate embedding for a table with a summary description.

        Args:
            df: Pandas DataFrame
            summary: Summary or description of the table

        Returns:
            Numpy array containing the embedding
        """
        # Convert table to text
        table_text = self.table_to_text(df)

        # Combine summary and table
        if summary:
            combined_text = f"{summary}\n\n{table_text}"
        else:
            combined_text = table_text

        # Generate embedding
        embedding = self.text_embedder.embed_text(combined_text)
        return embedding

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.

        Returns:
            Embedding dimension
        """
        return self.embedding_dim


if __name__ == "__main__":
    # Example usage
    embedder = TableEmbedder(table_format="markdown")

    # Create sample table
    sample_data = {
        "Product": ["Apple", "Banana", "Orange"],
        "Price": [1.20, 0.50, 0.80],
        "Quantity": [100, 150, 120],
    }
    df = pd.DataFrame(sample_data)

    print("Sample table:")
    print(df)

    # Convert to text
    table_text = embedder.table_to_text(df)
    print(f"\nTable as {embedder.table_format}:")
    print(table_text)

    # Generate embedding
    embedding = embedder.embed_table(df)
    print(f"\nTable embedding shape: {embedding.shape}")
    print(f"Embedding preview: {embedding[:5]}")

    # Embed with context
    embedding_with_context = embedder.embed_table(
        df, add_context="Table showing fruit prices and inventory:"
    )
    print(f"\nEmbedding with context shape: {embedding_with_context.shape}")

    # Multiple tables
    df2 = pd.DataFrame(
        {"Name": ["John", "Jane"], "Age": [30, 25], "City": ["NYC", "LA"]}
    )

    embeddings = embedder.embed_tables([df, df2])
    print(f"\nMultiple tables embeddings shape: {embeddings.shape}")

    # Embed table data with metadata
    table_data = [
        {"dataframe": df, "page_num": 0, "table_num": 0},
        {"dataframe": df2, "page_num": 1, "table_num": 0},
    ]

    embedded_data = embedder.embed_table_data(table_data)
    print(f"\nEmbedded {len(embedded_data)} tables with metadata")
    print(f"First table keys: {embedded_data[0].keys()}")
