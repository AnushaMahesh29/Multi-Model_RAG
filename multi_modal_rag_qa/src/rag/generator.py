"""
Generator Module
Generates answers using LLM (Groq API with LLaMA) based on retrieved context.
"""

from groq import Groq
from typing import Optional, Dict, Any
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Generator:
    """
    Generates answers using Groq API (LLaMA models).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        temperature: float = 1.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        stream: bool = False,
    ):
        """
        Initialize the generator.

        Args:
            api_key: Groq API key (reads from GROQ_API_KEY env var if None)
            model: Model name (e.g., 'meta-llama/llama-4-scout-17b-16e-instruct')
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            top_p: Top-p sampling parameter
            stream: Whether to use streaming responses
        """
        # Get API key
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stream = stream

        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)

        logger.info(f"Generator initialized with model: {model}")

    def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate an answer based on query and context.

        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional system prompt (uses default if None)

        Returns:
            Dictionary with answer and metadata
        """
        # Default system prompt
        if system_prompt is None:
            system_prompt = (
                "You are a helpful AI assistant that answers questions based on the provided context. "
                "Use the context to provide accurate, detailed answers. "
                "If the context doesn't contain enough information to answer the question, "
                "say so clearly. Cite specific parts of the context when possible."
            )

        # Construct user message
        user_message = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        try:
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                top_p=self.top_p,
                stream=self.stream,
                stop=None,
            )

            # Handle streaming vs non-streaming
            if self.stream:
                # Collect streamed response
                answer = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        answer += chunk.choices[0].delta.content
                
                result = {
                    "answer": answer,
                    "model": self.model,
                    "usage": {},  # Streaming doesn't provide usage stats immediately
                    "finish_reason": "stop",
                }
            else:
                # Extract answer from non-streaming response
                answer = response.choices[0].message.content

                result = {
                    "answer": answer,
                    "model": self.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    "finish_reason": response.choices[0].finish_reason,
                }

            logger.info(f"Generated answer using {self.model}")
            return result

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "model": self.model,
                "error": str(e),
            }

    def generate_simple(self, query: str, context: str) -> str:
        """
        Generate an answer and return just the text.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Answer text
        """
        result = self.generate(query, context)
        return result.get("answer", "")

    def generate_with_citations(
        self, query: str, context: str
    ) -> Dict[str, Any]:
        """
        Generate an answer with explicit citation instructions.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Dictionary with answer and metadata
        """
        system_prompt = (
            "You are a helpful AI assistant that answers questions based on the provided context. "
            "Always cite the specific parts of the context you use in your answer. "
            "Use references like [Page X] or [Table Y] when citing. "
            "If the context doesn't contain enough information, say so clearly."
        )

        return self.generate(query, context, system_prompt)

    def generate_concise(self, query: str, context: str) -> Dict[str, Any]:
        """
        Generate a concise answer.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Dictionary with answer and metadata
        """
        system_prompt = (
            "You are a helpful AI assistant that provides concise, direct answers. "
            "Answer the question based on the context in 2-3 sentences maximum. "
            "Be accurate and to the point."
        )

        return self.generate(query, context, system_prompt)


if __name__ == "__main__":
    # Example usage
    # Note: Requires GROQ_API_KEY environment variable to be set

    try:
        generator = Generator(model="meta-llama/llama-4-scout-17b-16e-instruct")

        # Sample context and query
        sample_context = """
        [Text from Page 0, Chunk 0]
        Machine learning is a subset of artificial intelligence that enables 
        computers to learn from data without being explicitly programmed.
        
        [Table from Page 1, Table 0]
        | Model | Accuracy |
        |-------|----------|
        | Model A | 95% |
        | Model B | 92% |
        """

        sample_query = "What is the accuracy of Model A?"

        # Generate answer
        result = generator.generate(sample_query, sample_context)

        print("Query:", sample_query)
        print("\nAnswer:", result["answer"])
        print("\nTokens used:", result["usage"]["total_tokens"])

        # Generate with citations
        print("\n" + "=" * 50)
        result_citations = generator.generate_with_citations(
            sample_query, sample_context
        )
        print("\nAnswer with citations:", result_citations["answer"])

    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo use the generator, set your Groq API key:")
        print("  export GROQ_API_KEY='your-api-key-here'")
        print("\nGet your API key at: https://console.groq.com/")
