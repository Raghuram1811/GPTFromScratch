"""
===============================================================================
Self-Attention Module
===============================================================================

This module demonstrates scaled dot-product self-attention, the core operation
inside GPT-style transformer blocks.

Key Concepts:
- Query: What each token is looking for.
- Key: What each token offers for matching.
- Value: The information each token contributes.
- Attention Scores: Similarity between every query and every key.
- Attention Weights: Softmax-normalized scores.
- Context Vectors: Weighted sums of the value vectors.

Input/Output Shapes:
- Input embeddings:  (batch_size, context_length, embedding_dim)
- Attention scores:  (batch_size, context_length, context_length)
- Attention weights: (batch_size, context_length, context_length)
- Output context:    (batch_size, context_length, output_dim)

Author: GPTFromScratch
Date: 2026-05-11
===============================================================================
"""

import os
import sys
from typing import Tuple

import torch
from torch import nn

# ============================================================================
# Setup Project Path
# ============================================================================
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessor.data.prepare_data import DataPreProcessor, TorchWrapper


# ============================================================================
# Configuration Constants
# ============================================================================
VOCAB_SIZE = 50257
CONTEXT_LENGTH = 5
BATCH_SIZE = 2
STRIDE = 2
EMBEDDING_DIM = 16
OUTPUT_DIM = 16


# ============================================================================
# Self-Attention Layer
# ============================================================================
class SelfAttention(nn.Module):
    """
    Single-head scaled dot-product self-attention.

    Args:
        input_dim: Size of each input token embedding.
        output_dim: Size of the query, key, value, and output vectors.
        use_bias: Whether linear projections should include bias terms.

    Example:
        >>> attention = SelfAttention(input_dim=768, output_dim=768)
        >>> x = torch.randn(4, 128, 768)
        >>> context, weights = attention(x)
        >>> context.shape
        torch.Size([4, 128, 768])
    """

    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = False) -> None:
        super().__init__()
        self.output_dim = output_dim

        self.query = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.key = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.value = nn.Linear(input_dim, output_dim, bias=use_bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute self-attention for a batch of token embeddings.

        Args:
            x: Tensor with shape (batch_size, context_length, input_dim).

        Returns:
            A tuple containing:
            - context_vectors: Tensor with shape
              (batch_size, context_length, output_dim).
            - attention_weights: Tensor with shape
              (batch_size, context_length, context_length).
        """
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        attention_scores = queries @ keys.transpose(-2, -1) # Shape: (batch_size, context_length, context_length), (-2, -1) means we transpose the last two dimensions of keys for proper matrix multiplication
        attention_scores = attention_scores / (self.output_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1) # dim=-1 means we apply softmax across the last dimension (context_length) to get attention weights that sum to 1 for each query token

        context_vectors = attention_weights @ values
        return context_vectors, attention_weights


# ============================================================================
# Helper Functions
# ============================================================================
def build_sample_embeddings() -> torch.Tensor:
    """
    Create a small batch of token plus positional embeddings for demonstration.

    Returns:
        Tensor with shape (BATCH_SIZE, CONTEXT_LENGTH, EMBEDDING_DIM).
    """
    text = """
    Self-attention lets each token decide which other tokens matter for building
    its next representation. This simple example prepares token embeddings and
    sends them through one attention layer.
    """

    dataset = DataPreProcessor(
        text=text,
        model="gpt2",
        context_length=CONTEXT_LENGTH,
        stride=STRIDE,
    )
    torch_wrapper = TorchWrapper(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    token_ids = next(iter(torch_wrapper.dataloader))
    token_embedding_layer = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
    positional_embedding_layer = nn.Embedding(CONTEXT_LENGTH, EMBEDDING_DIM)

    token_embeddings = token_embedding_layer(token_ids)
    positions = torch.arange(CONTEXT_LENGTH)
    positional_embeddings = positional_embedding_layer(positions).unsqueeze(0)

    print(f"Token id batch shape:        {token_ids.shape}")
    print(f"Token embeddings shape:      {token_embeddings.shape}")
    print(f"Positional embeddings shape: {positional_embeddings.shape}")

    return token_embeddings + positional_embeddings


# ============================================================================
# Main Function
# ============================================================================
def main() -> None:
    """Run a small end-to-end self-attention demonstration."""
    torch.manual_seed(123)

    print("\n" + "=" * 80)
    print("STEP 1: CREATE INPUT EMBEDDINGS")
    print("=" * 80)
    input_embeddings = build_sample_embeddings()
    print(f"Combined input shape:        {input_embeddings.shape}\n")

    print("=" * 80)
    print("STEP 2: APPLY SELF-ATTENTION")
    print("=" * 80)
    attention_layer = SelfAttention(
        input_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        use_bias=False,
    )

    context_vectors, attention_weights = attention_layer(input_embeddings)

    print(f"Attention weights shape:     {attention_weights.shape}")
    print(f"Context vectors shape:       {context_vectors.shape}\n")

    print("=" * 80)
    print("STEP 3: INSPECT ONE TOKEN")
    print("=" * 80)
    print("Attention weights for batch 0, token 0:")
    print(attention_weights[0, 0])
    print("\nContext vector for batch 0, token 0:")
    print(context_vectors[0, 0])


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    main()
