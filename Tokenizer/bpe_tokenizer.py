"""
================================================================================
Byte Pair Encoding (BPE) Tokenizer Implementation
================================================================================
Author: Raghuram1811
Date: 2026-05-02
Description: This module implements the Byte Pair Encoding (BPE) tokenization
algorithm, which is commonly used in natural language processing and large
language models (like GPT). BPE works by iteratively merging the most frequent
pairs of tokens to reduce vocabulary size and improve compression.

References:
- BPE Paper: https://arxiv.org/abs/1508.07909
- Originally developed for building basic GPT model.

"""

import math


class BPETokenizer():
    """
    Byte Pair Encoding (BPE) Tokenizer class.
    
    This class implements the BPE tokenization algorithm which:
    1. Starts with individual bytes (0-255)
    2. Finds the most frequent pair of tokens
    3. Merges that pair into a new token
    4. Repeats until convergence or a stopping condition
    
    Attributes:
        merges (dict): Maps token pairs to their merged token ID
        next_id (int): Counter for new token IDs (starts at 256)
        vocab (dict): Maps token IDs to their byte representations
    """

    def __init__(self):
        """
        Initialize the BPETokenizer with base vocabulary.
        
        Sets up:
        - merges: Empty dictionary to track pair->id mappings
        - next_id: Starts at 256 (after all single bytes 0-255)
        - vocab: Maps bytes 0-255 to their byte representations
        """
        self.merges = dict()
        self.next_id = 256
        self.vocab = {i: bytes([i]) for i in range(0, 256)}
    
    def rank_pairs(self, token_ids: list):
        """
        Count the frequency of all adjacent token pairs.
        
        Args:
            token_ids (list): List of token IDs to analyze
            
        Returns:
            dict: Dictionary mapping token pairs (tuples) to their frequencies
            
        Example:
            >>> tokenizer = BPETokenizer()
            >>> pairs = tokenizer.rank_pairs([1, 2, 1, 2, 3])
            >>> # Returns {(1, 2): 2, (2, 1): 1, (2, 3): 1}
        """
        counts = {}
        # Iterate through adjacent pairs
        for idx in range(len(token_ids) - 1):
            pair = (token_ids[idx], token_ids[idx + 1])
            # Increment count for this pair (default to 0 if not seen before)
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge_tokens(self, new_id, pair, token_ids):
        """
        Replace all occurrences of a token pair with a new merged token ID.
        
        Args:
            new_id (int): The new token ID to use for the merged pair
            pair (tuple): The token pair (old_id_1, old_id_2) to merge
            token_ids (list): The current list of token IDs
            
        Returns:
            list: New list of token IDs with the pair merged
            
        Example:
            >>> tokenizer = BPETokenizer()
            >>> result = tokenizer.merge_tokens(256, (1, 2), [1, 2, 3, 1, 2])
            >>> # Returns [256, 3, 256]
        """
        new_tokens, idx = [], 0
        
        # Scan through token_ids looking for the pair to merge
        while idx < len(token_ids):
            # Check if current position matches the pair (and next token exists)
            if idx < len(token_ids) - 1 and pair == (token_ids[idx], token_ids[idx + 1]):
                # Found the pair - replace with new merged token
                new_tokens.append(new_id)
                idx += 2  # Skip both tokens in the pair
            else:
                # Current token doesn't match the pair - keep it as is
                new_tokens.append(token_ids[idx])
                idx += 1
        
        return new_tokens

    def tokenize(self, b_array: list[int], top_frequency: float = math.inf):
        """
        Perform Byte Pair Encoding tokenization on a list of byte values.
        
        This method recursively:
        1. Finds the most frequent adjacent token pair
        2. Creates a new token ID for this pair
        3. Merges all occurrences of the pair
        4. Repeats until stopping condition is met
        
        Args:
            b_array (list[int]): List of byte values (0-255) or previously merged token IDs
            top_frequency (float): Stopping condition - stop when max pair frequency <= this value
                                   Defaults to math.inf (merge until only 1 unique pair remains)
        
        Returns:
            list: The tokenized/encoded representation of the input
            
        Example:
            >>> tokenizer = BPETokenizer()
            >>> input_bytes = list(b"hello")
            >>> result = tokenizer.tokenize(input_bytes, top_frequency=2)
        """
        # Base case: if top_frequency threshold reached, stop merging
        if top_frequency <= 1:
            return b_array

        # Count frequency of all adjacent token pairs
        pair_counts = self.rank_pairs(b_array)

        # Base case: no pairs to merge (single token or empty)
        if not pair_counts:
            return b_array

        # Find the most frequently occurring pair
        top_pair, top_frequency = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[0]

        # Assign a new ID to this pair
        id = self.next_id
        self.next_id += 1

        # Record this merge for future reference
        self.merges[top_pair] = id
        
        # Store the vocabulary entry: new token = concatenation of pair components
        self.vocab[id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

        # Replace all occurrences of the pair with the new token ID
        b_array = self.merge_tokens(id, top_pair, b_array)

        # Recursively continue merging with updated token list
        return self.tokenize(b_array=b_array, top_frequency=top_frequency)


def main():
    """
    Main function demonstrating BPE tokenizer usage.
    
    Creates a sample text with repeated words, encodes it to UTF-8 bytes,
    performs BPE tokenization, and prints the resulting vocabulary.
    """
    # Sample text with repeated patterns (good for BPE compression)
    input_string = """ant ant ant ant ant ant ants ants ants plant plant plants gigantic gigantic gigantic"""
    
    # Convert string to UTF-8 byte representation
    encoded_string = input_string.encode('utf-8')
    
    # Initialize tokenizer and perform BPE
    bpe_tokenizer = BPETokenizer()
    bpe_tokenizer.tokenize(b_array=list(encoded_string))

    # Optional: Print merge operations (commented out for cleaner output)
    # for pair, new_id in bpe_tokenizer.merges.items():
    #     print(pair, "=>", new_id, bpe_tokenizer.vocab[new_id])
    
    # Print the complete vocabulary after BPE tokenization
    # Shows all token IDs and their corresponding byte representations
    for item in bpe_tokenizer.vocab.items():
        print(item)


if __name__ == '__main__':
    main()
