"""LLM-based voting implementation for Co-Forget Protocol."""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np


@dataclass
class LLMVoter:
    """Uses LLM for memory voting decisions."""

    model_name: str = "distilbert-base-uncased"
    relevance_threshold: float = 0.7
    max_length: int = 512

    def __post_init__(self):
        """Initialize the LLM model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.classifier = pipeline(
            "text-classification", model=self.model_name, tokenizer=self.tokenizer
        )

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get BERT embedding for text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embedding

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)

        # Normalize embeddings
        emb1 = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
        emb2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)

        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2.T)[0][0]
        return float(similarity)

    def _check_relevance(
        self, memory_text: str, current_context: str
    ) -> Tuple[bool, float]:
        """Check if memory is relevant to current context."""
        # Calculate similarity
        similarity = self._calculate_similarity(memory_text, current_context)

        # Use classifier for additional context
        prompt = f"Memory: {memory_text}\nContext: {current_context}\nIs this memory relevant?"
        result = self.classifier(prompt)[0]

        # Combine similarity and classification
        relevance_score = (similarity + float(result["score"])) / 2
        is_relevant = relevance_score >= self.relevance_threshold

        return is_relevant, relevance_score

    def vote(
        self,
        memory_text: str,
        memory_metadata: Dict,
        current_context: str,
        decay_score: float,
    ) -> Tuple[str, float]:
        """Vote on whether to forget a memory.

        Returns:
            Tuple[str, float]: (vote, confidence)
            vote is either "forget" or "keep"
            confidence is a float between 0 and 1
        """
        # If decay score is very low, forget without LLM check
        if decay_score < 0.1:
            return "forget", 1.0

        # If decay score is very high, keep without LLM check
        if decay_score > 0.9:
            return "keep", 1.0

        # For borderline cases, use LLM
        is_relevant, relevance_score = self._check_relevance(
            memory_text, current_context
        )

        # Combine decay and relevance scores
        if is_relevant:
            # If relevant, bias towards keeping
            final_score = (decay_score + relevance_score) / 2
            vote = "keep" if final_score > 0.5 else "forget"
        else:
            # If not relevant, bias towards forgetting
            final_score = (decay_score + (1 - relevance_score)) / 2
            vote = "forget" if final_score < 0.5 else "keep"

        return vote, abs(final_score - 0.5) * 2  # Convert to confidence score
