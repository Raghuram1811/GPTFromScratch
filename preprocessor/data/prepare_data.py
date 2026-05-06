"""
Data Preparation Module for GPT Model Training

This module provides utilities for preprocessing raw text data into tokenized sequences
suitable for training a GPT-style language model. It handles tokenization, sequence
creation with configurable context length and stride, and PyTorch DataLoader integration.

Key Components:
    - DataPreProcessor: PyTorch Dataset class for creating input-target token sequences
    - TorchWrapper: Wrapper around PyTorch DataLoader for batch processing
"""

import os
import sys
import torch

# Setup project root path for module imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessor.Tokenizer.tiktokenizer import TikTokenizer
from torch.utils.data import DataLoader, Dataset


class DataPreProcessor(Dataset):
    """
    PyTorch Dataset class for preprocessing text data into token sequences.
    
    Converts raw text into tokenized sequences where each sequence consists of
    an input (context) and target (next token prediction) pair. This is the
    standard format for language model pretraining.
    
    Args:
        text (str): Raw text data to tokenize and process.
        model (str): Model identifier for the tokenizer (e.g., 'gpt2'). This specifies
                     which tokenizer vocabulary to use.
        context_length (int): Length of context window for each input sequence.
                             Defines how many tokens form a single input example.
        stride (int): Step size for sliding window. Controls overlap between consecutive
                     sequences. Smaller stride = more overlapping sequences.
    
    Attributes:
        input (list): List of input token sequences (context).
        target (list): List of target token sequences (next token predictions).
    
    Example:
        >>> dataset = DataPreProcessor(
        ...     text="Hello world",
        ...     model="gpt2",
        ...     context_length=4,
        ...     stride=1
        ... )
        >>> len(dataset)  # Number of sequences created
        >>> input_tokens, target_tokens = dataset[0]
    """

    def __init__(self, text, model, context_length, stride):
        """Initialize the dataset by tokenizing and preparing sequences."""
        self.input = []
        self.target = []

        # Initialize tokenizer with specified model
        tiktokenizer = TikTokenizer(model=model)
        # Encode entire text into token IDs, preserving special end-of-text tokens
        token_ids = tiktokenizer.encode(input=text, allowed_special={"<|endoftext|>"})
        
        # Create sliding window sequences for input-target pairs
        # Each iteration creates a sequence where:
        # - input: tokens[i:i+context_length]
        # - target: tokens[i+1:i+context_length+1] (shifted by 1 for next-token prediction)
        for i in range(0, len(token_ids) - context_length, stride):
            # Extract context window as model input
            input_row = token_ids[i:i + context_length]
            # Extract shifted window as prediction target (next token prediction task)
            target_row = token_ids[i + 1:i + context_length + 1]

            # Convert to PyTorch tensors for GPU compatibility
            self.input.append(torch.tensor(input_row))
            self.target.append(torch.tensor(target_row))
        
        # Debugging info (uncomment for sequence inspection):
        # print(f"Total input sequences: {len(self.input)}")
        # print(f"Total target sequences: {len(self.target)}")
    
    def __len__(self):
        """Return the total number of input-target pairs in the dataset."""
        return len(self.input)

    def __getitem__(self, idx):
        """
        Retrieve a single input-target pair by index.
        
        Args:
            idx (int): Index of the sequence to retrieve.
        
        Returns:
            tuple: (input_tensor, target_tensor) for the given index.
        """
        return self.input[idx], self.target[idx]
    
    def __gettensors__(self, idx):
        """
        Alternative method to retrieve tensors (ensures tensor conversion).
        
        Note: This method is redundant since __getitem__ already returns tensors.
              Consider removing or documenting the specific use case.
        
        Args:
            idx (int): Index of the sequence to retrieve.
        
        Returns:
            tuple: (input_tensor, target_tensor) as PyTorch tensors.
        """
        return torch.tensor(self.input[idx]), torch.tensor(self.target[idx])


class TorchWrapper:
    """
    Wrapper class for PyTorch DataLoader.
    
    Simplifies batch processing and data iteration by wrapping PyTorch's DataLoader.
    Handles shuffling, batching, and multi-worker data loading for efficient training.
    
    Args:
        dataset (Dataset): PyTorch Dataset object containing the data.
        batch_size (int): Number of samples per batch.
        shuffle (bool, optional): Whether to shuffle data during iteration.
                                 Default: True.
    
    Attributes:
        dataloader (DataLoader): PyTorch DataLoader instance managing batching.
    
    Example:
        >>> dataset = DataPreProcessor(text, model="gpt2", context_length=4, stride=1)
        >>> wrapper = TorchWrapper(dataset, batch_size=32, shuffle=True)
        >>> for batch_input, batch_target in wrapper.dataloader:
        ...     # Process batch
        ...     pass
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        """
        Initialize the DataLoader wrapper.
        
        Args:
            dataset (Dataset): The dataset to wrap.
            batch_size (int): Batch size for iteration.
            shuffle (bool): Whether to shuffle batches. Default: True.
        """
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,  # Drop incomplete batches for consistent batch sizes
            num_workers=2    # Use 2 worker processes for data loading
        )


def main():
    """
    Demonstration and testing of the data preprocessing pipeline.
    
    This function:
    1. Creates sample text data (Tendulkar biography excerpt)
    2. Initializes DataPreProcessor with GPT-2 tokenizer
    3. Wraps the dataset with TorchWrapper for batch loading
    4. Can iterate through batches for verification (see commented code)
    """
    # Sample text data for processing
    text = """Tendulkar took up cricket at the age of eleven, made his Test match debut on 15 November 1989 against Pakistan in Karachi at the age of sixteen, and went on to represent Mumbai domestica[...]

Tendulkar has received several awards from the government of India: the Arjuna Award (1994), the Khel Ratna Award (1997), the Padma Shri (1998), and the Padma Vibhushan (2008).[14][15] After Tendulkar[...]

Tendulkar is regarded as a symbol of national pride in India for his achievements. In 2010, Time included Tendulkar in its annual list of the most influential people in the world.[26] Tendulkar was aw[...]
    """
    
    # Create dataset with small context window and stride for demonstration
    dataset = DataPreProcessor(text, model="gpt2", context_length=4, stride=4)
    # Wrap dataset for batch iteration
    torch_wrapper = TorchWrapper(dataset=dataset, batch_size=20, shuffle=False)
    
    # Uncomment below to verify data loading and inspect sample batches:
    # print("DataLoader created successfully. Sample batches:")
    # for idx, batch in enumerate(torch_wrapper.dataloader):
    #     print(f"Batch {idx+1}: {batch}")


if __name__ == '__main__':
    main()
