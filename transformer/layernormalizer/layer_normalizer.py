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


class LayerNormalizer(nn.Module):

    def __init__(self, input: torch.Tensor):
        self.input = input
        self.eps = 1e-5
        super(LayerNormalizer, self).__init__()
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.input.mean(dim=-1, keepdim=True) # -1 means we calculate mean across the columns of the input matrix
        variance = self.input.var(dim=-1, keepdim=True) # -1 means we calculate variance across the columns of the input matrix
        normalized_input = (self.input - mean)/((variance + self.eps)** 0.5) # Formulaic implementation of standardizer (x-mean/standard_deviation); sqrt(variance) gives standard deviation - we add self.eps to ensure a fraction value is added incase variance moves to 0. This ensures we avoid dividing by 0.
        return normalized_input

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

    print("="*80)
    print(f"Implementing Layer Normalizer here!")
    input_embeddings = build_sample_embeddings()
    print(f"Combined input shape:        {input_embeddings.shape}\n")

    layer_normalizer = LayerNormalizer(input_embeddings)
    normalized_input = layer_normalizer() # Executes the forward implementation of LayerNormalizer
    print(normalized_input.mean(dim=-1), normalized_input.var(dim=-1) )
    

if __name__ == '__main__':
    main()
