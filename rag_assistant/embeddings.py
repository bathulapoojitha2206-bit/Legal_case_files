"""
Hugging Face embeddings manager for legal text vectorization.
"""

import logging
from typing import Optional
import torch
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages Hugging Face sentence-transformers embeddings for FAISS indexing."""

    _instance: Optional[HuggingFaceEmbeddings] = None
    _current_model_name: Optional[str] = None
    _current_device: Optional[str] = None

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        self.model_name = model_name

        # Auto-detect device if requested or if cuda is specified but unavailable
        if device.lower() == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = device.lower()

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Return the cached or newly initialized HuggingFaceEmbeddings instance.
        """
        if (
            EmbeddingManager._instance is not None
            and EmbeddingManager._current_model_name == self.model_name
            and EmbeddingManager._current_device == self.device
        ):
            return EmbeddingManager._instance

        logger.info(
            "Loading HuggingFace embeddings model: %s on device: %s",
            self.model_name,
            self.device
        )

        encode_kwargs = {"normalize_embeddings": True}
        model_kwargs = {"device": self.device}

        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            EmbeddingManager._instance = embeddings
            EmbeddingManager._current_model_name = self.model_name
            EmbeddingManager._current_device = self.device
            return embeddings
        except Exception as exc:
            logger.error("Failed to initialize HuggingFace embeddings: %s", exc)
            raise RuntimeError(
                f"Failed to load Hugging Face embeddings model '{self.model_name}'. "
                f"Ensure internet connectivity or check local HuggingFace cache. Details: {exc}"
            ) from exc
