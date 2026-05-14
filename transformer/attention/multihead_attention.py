

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

class MultiHeadAttentionIllustrated(nn.Module):
    """
    Multi-head attention layer that combines multiple self-attention heads.

    Args:
        input_dim: Size of each input token embedding.
        output_dim: Size of the query, key, value, and output vectors for each hea
        
        d.
        num_heads: Number of attention heads to use.
        use_bias: Whether linear projections should include bias terms.

    Example:
        >>> multi_head_attention = MultiHeadAttentionIllustrated(input_dim=768, output_dim=768, num_heads=8)
        >>> x = torch.randn(4, 128, 768)
        >>> context, weights = multi_head_attention(x)
        >>> context.shape
        torch.Size([4, 128, 768])
    """
    def __init__(self, input_dim: int, output_dim: int, num_heads: int, use_bias: bool = False):
        super(MultiHeadAttentionIllustrated, self).__init__()

        assert output_dim % num_heads == 0, "Output dimension must be divisible by the number of heads."
        
        self.num_heads = num_heads
        self.output_dim = output_dim

        self.head_dim = self.output_dim // self.num_heads

        self.query_proj = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.key_proj = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.value_proj = nn.Linear(input_dim, output_dim, bias=use_bias)
        
        self.out_proj = nn.Linear(output_dim, output_dim)  # Final linear layer to combine heads

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, context_length, _ = x.size()
        queries = self.query_proj(x).view(batch_size, context_length, self.num_heads, -1).transpose(1, 2)  # Shape: (batch_size, num_heads, context_length, head_dim)
        keys = self.key_proj(x).view(batch_size, context_length, self.num_heads, -1).transpose(1, 2)  # Shape: (batch_size, num_heads, context_length, head_dim) 
        values = self.value_proj(x).view(batch_size, context_length, self.num_heads, -1).transpose(1, 2)  # Shape: (batch_size, num_heads, context_length, head_dim)

        attention_scores = queries @ keys.transpose(-2, -1)  # Shape: (batch_size, num_heads, context_length, context_length)
        attention_scores = attention_scores / (self.output_dim // self.num_heads) ** 0.5 # divide with sqrt of head dimensions
        attention_weights = torch.softmax(attention_scores, dim=-1) # Shape: (batch_size, num_heads, context_length, context_length)
        context_vectors = attention_weights @ values  # Shape: (batch_size, num_heads, context_length, head_dim)
        context_vectors = context_vectors.transpose(1, 2).contiguous().view(batch_size, context_length, self.output_dim)  # Shape: (batch_size, context_length,output_dim)

        output = self.out_proj(context_vectors)  # Shape: (batch_size, context_length, output_dim)
        return output, attention_weights
    

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

    token_ids = next(iter(torch_wrapper.dataloader)) # next(iter(...)) is a common way to get the first batch from a DataLoader. It returns a tuple of (input_tensor, target_tensor) for the first batch. Since we only need the input tensor for building embeddings, we can unpack it directly.
    token_embedding_layer = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
    positional_embedding_layer = nn.Embedding(CONTEXT_LENGTH, EMBEDDING_DIM)

    token_embeddings = token_embedding_layer(token_ids)
    positions = torch.arange(CONTEXT_LENGTH)
    positional_embeddings = positional_embedding_layer(positions).unsqueeze(0)

    print(f"Token id batch shape:        {token_ids.shape}")
    print(f"Token embeddings shape:      {token_embeddings.shape}")
    print(f"Positional embeddings shape: {positional_embeddings.shape}")

    return token_embeddings + positional_embeddings

def main():
    """Run a small multi-head attention demonstration."""
    torch.manual_seed(123)

    print("\n" + "=" * 80)
    print("STEP 1: BUILD SAMPLE EMBEDDINGS")
    print("=" * 80)
    embeddings = build_sample_embeddings()
    print(f"Combined input shape:        {embeddings.shape}\n")

    print("=" * 80)
    print("STEP 2: APPLY MULTI-HEAD-ATTENTION")
    print("=" * 80)
    attention_layer = MultiHeadAttentionIllustrated(
        input_dim=EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        num_heads=2,
        use_bias=False,
    )
    context_vectors, attention_weights = attention_layer(embeddings)
    print(f"Context shape:               {context_vectors.shape}")
    print(f"Attention weights shape:     {attention_weights.shape}")
    print("=" * 80)
    print("STEP 3: INSPECT ONE TOKEN")
    print("=" * 80)
    print("Attention weights for batch 0, token 0:")
    print(attention_weights[0, 0])
    print("\nContext vector for batch 0, token 0:")
    print(context_vectors[0, 0])

if __name__ == "__main__":
    main()