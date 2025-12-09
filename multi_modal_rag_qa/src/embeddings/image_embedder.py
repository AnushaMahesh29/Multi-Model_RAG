"""
Image Embedder Module
Generates embeddings for images using CLIP.
"""

import torch
import clip
from PIL import Image
from typing import List, Dict, Any, Union
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageEmbedder:
    """
    Generates embeddings for images using CLIP model.
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize the image embedder.

        Args:
            model_name: CLIP model name ('ViT-B/32', 'ViT-B/16', 'ViT-L/14')
            device: Device to run the model on (None for auto-detect)
        """
        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading CLIP model: {model_name} on {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        # Get embedding dimension
        self.embedding_dim = self.model.visual.output_dim
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed_image(self, image: Union[str, Image.Image]) -> np.ndarray:
        """
        Generate embedding for a single image.

        Args:
            image: PIL Image object or path to image file

        Returns:
            Numpy array containing the embedding
        """
        try:
            # Load image if path is provided
            if isinstance(image, (str, Path)):
                image = Image.open(image)

            # Preprocess and encode
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Convert to numpy
            embedding = image_features.cpu().numpy().squeeze()
            return embedding

        except Exception as e:
            logger.error(f"Error embedding image: {e}")
            # Return zero vector on error
            return np.zeros(self.embedding_dim)

    def embed_images(self, images: List[Union[str, Image.Image]], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple images.

        Args:
            images: List of PIL Image objects or paths to image files
            batch_size: Batch size for encoding

        Returns:
            Numpy array of shape (n_images, embedding_dim)
        """
        if not images:
            return np.array([])

        all_embeddings = []

        logger.info(f"Embedding {len(images)} images...")

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            # Load and preprocess batch
            batch_images = []
            for img in batch:
                try:
                    if isinstance(img, (str, Path)):
                        img = Image.open(img)
                    batch_images.append(self.preprocess(img))
                except Exception as e:
                    logger.error(f"Error loading image: {e}")
                    # Add zero tensor for failed images
                    batch_images.append(torch.zeros(3, 224, 224))

            # Stack and move to device
            batch_tensor = torch.stack(batch_images).to(self.device)

            # Encode
            with torch.no_grad():
                image_features = self.model.encode_image(batch_tensor)
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Convert to numpy and append
            embeddings = image_features.cpu().numpy()
            all_embeddings.append(embeddings)

        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)
        logger.info(f"Generated embeddings with shape: {all_embeddings.shape}")

        return all_embeddings

    def embed_image_data(
        self, image_data: List[Dict[str, Any]], batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for image data with metadata.

        Args:
            image_data: List of dictionaries with 'saved_path' or 'pil_image' key
            batch_size: Batch size for encoding

        Returns:
            List of dictionaries with added 'embedding' key
        """
        if not image_data:
            return []

        # Extract images
        images = []
        for data in image_data:
            if "saved_path" in data:
                images.append(data["saved_path"])
            elif "pil_image" in data:
                images.append(data["pil_image"])
            else:
                logger.warning("Image data missing 'saved_path' or 'pil_image' key")
                images.append(None)

        # Generate embeddings
        embeddings = []
        for img in images:
            if img is not None:
                embedding = self.embed_image(img)
            else:
                embedding = np.zeros(self.embedding_dim)
            embeddings.append(embedding)

        embeddings = np.array(embeddings)

        # Add embeddings to data
        for data, embedding in zip(image_data, embeddings):
            data["embedding"] = embedding
            data["embedding_model"] = self.model_name

        logger.info(f"Added embeddings to {len(image_data)} images")
        return image_data

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
        # CLIP embeddings are already normalized, so dot product = cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)


if __name__ == "__main__":
    # Example usage
    embedder = ImageEmbedder(model_name="ViT-B/32")

    print(f"Embedding dimension: {embedder.get_embedding_dimension()}")

    # Note: This example requires actual image files to run
    # Uncomment and modify paths as needed

    # # Single image
    # image_path = "data/intermediate/images/page_0_img_0.png"
    # if Path(image_path).exists():
    #     embedding = embedder.embed_image(image_path)
    #     print(f"Single image embedding shape: {embedding.shape}")
    #     print(f"Embedding preview: {embedding[:5]}")

    # # Multiple images
    # image_dir = Path("data/intermediate/images")
    # if image_dir.exists():
    #     image_paths = list(image_dir.glob("*.png"))[:3]
    #     if image_paths:
    #         embeddings = embedder.embed_images(image_paths)
    #         print(f"\nMultiple images embeddings shape: {embeddings.shape}")

    #         # Compute similarity
    #         if len(embeddings) >= 2:
    #             similarity = embedder.compute_similarity(embeddings[0], embeddings[1])
    #             print(f"Similarity between image 0 and 1: {similarity:.4f}")

    print("\nImage embedder initialized successfully!")
    print("Add image files to test embedding functionality.")
