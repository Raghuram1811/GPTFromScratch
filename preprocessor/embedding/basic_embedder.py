"""
===============================================================================
Basic Embedder Module
===============================================================================

This module demonstrates token embeddings and positional embeddings for a 
GPT-2 style language model. It shows how to:
1. Create token embeddings (lookup table mapping token IDs to dense vectors)
2. Create positional embeddings (encoding token positions in sequences)
3. Combine token and positional embeddings for input to transformer layers

Key Concepts:
- Vocabulary Size: Total number of unique tokens in the dataset (50,257 for GPT-2)
- Embedding Dimension: Size of each embedding vector (768 for demonstration)
- Context Length: Maximum sequence length the model can process
- Token Embeddings: Maps token IDs to semantic vector representations
- Positional Embeddings: Adds information about token positions in sequence

Input/Output Shapes:
- Batch input shape: (batch_size, context_length)
- Token embedding output: (batch_size, context_length, embedding_dim)
- Combined embedding output: (batch_size, context_length, embedding_dim)

Author: GPTFromScratch
Date: 2026-05-07
===============================================================================
"""

import torch
import os
import sys
from typing import Tuple

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
VOCAB_SIZE = 50257  # GPT-2 vocabulary size
EMBEDDING_DIM = 768  # Embedding dimension
CONTEXT_LENGTH = 5  # Maximum sequence length
BATCH_SIZE = 20  # Training batch size
STRIDE = 2  # Stride for sliding window in data preparation


# ============================================================================
# Main Function
# ============================================================================
def main() -> None:
    """
    Demonstrate token and positional embedding creation and combination.
    
    This function:
    1. Prepares sample text data using DataPreProcessor
    2. Creates token embedding layer (vocabulary lookup table)
    3. Creates positional embedding layer (position encoding)
    4. Combines both embeddings for each batch
    5. Prints embedding shapes and tensors for verification
    
    Returns:
        None
    """
    
    # ========================================================================
    # Step 1: Prepare Sample Data
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: DATA PREPARATION")
    print("="*80)
    
    text = """Sachin Ramesh Tendulkar is a former Indian international cricketer and a 
    former captain of the Indian national team. He is widely regarded as one of the 
    greatest batsmen in the history of cricket."""
    
    print(f"Text sample length: {len(text)} characters")
    print(f"Creating dataset with context_length={CONTEXT_LENGTH}, stride={STRIDE}")
    
    dataset = DataPreProcessor(
        text=text,
        model="gpt2",
        context_length=CONTEXT_LENGTH,
        stride=STRIDE
    )
    
    torch_wrapper = TorchWrapper(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Batch size: {BATCH_SIZE} \n")
    
    
    # ========================================================================
    # Step 2: Create Embedding Layers
    # ========================================================================
    print("="*80)
    print("STEP 2: CREATE EMBEDDING LAYERS")
    print("="*80)
    
    print(f"Token Embedding Layer:")
    print(f"  - Vocabulary size: {VOCAB_SIZE}")
    print(f"  - Embedding dimension: {EMBEDDING_DIM}")
    
    token_embedding_layer = torch.nn.Embedding(
        num_embeddings=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM
    )
    
    print(f"\nPositional Embedding Layer:")
    print(f"  - Max positions: {CONTEXT_LENGTH}")
    print(f"  - Embedding dimension: {EMBEDDING_DIM}\n")
    
    #  Positional embeddings initialized with max positions equal to context length, since we only need positional encodings for the maximum sequence length we will process. Each position (0 to CONTEXT_LENGTH-1) will have its own embedding vector of size EMBEDDING_DIM.
    positional_embedding_layer = torch.nn.Embedding(
        num_embeddings=CONTEXT_LENGTH,
        embedding_dim=EMBEDDING_DIM
    )
    
    # Pre-compute positional embeddings for all positions (0 to CONTEXT_LENGTH-1), example: for context_length=5, positions would be [0, 1, 2, 3, 4]
    positions = torch.arange(CONTEXT_LENGTH)
    positional_embeddings = positional_embedding_layer(positions)
    
    # Reshape for batch broadcasting: (context_length, embedding_dim) 
    # -> (1, context_length, embedding_dim) 
    # Reason for unsqueeze(0): to allow broadcasting across batch dimension, 0 means we add a new dimension at the front for batch size
    positional_embeddings = positional_embeddings.unsqueeze(0)
    
    print(f"Positional embeddings shape: {positional_embeddings.shape}")
    print(f"  -> Ready for broadcasting with batches\n")
    
    
    # ========================================================================
    # Step 3: Process Batches and Combine Embeddings
    # ========================================================================
    print("="*80)
    print("STEP 3: PROCESS BATCHES")
    print("="*80 + "\n")
    
    for batch_idx, batch in enumerate(torch_wrapper.dataloader, 1):
        print(f"Batch {batch_idx}:")
        print(f"  Input batch shape: {batch.shape}")
        print(f"  Input batch dtype: {batch.dtype}")
        
        # Get token embeddings
        token_embeddings = token_embedding_layer(batch)
        print(f"  Token embeddings shape: {token_embeddings.shape}")
        
        # Combine token and positional embeddings
        combined_embeddings = token_embeddings + positional_embeddings
        print(f"  Combined embeddings shape: {combined_embeddings.shape}")
        print(f"  Combined embeddings dtype: {combined_embeddings.dtype}")
        
        # Optional: Display sample embedding values
        print(f"  Sample embedding (first token of first sample):")
        print(f"    {combined_embeddings[0, 0, :5]}... (showing first 5 dims)")
        print()


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    main()
