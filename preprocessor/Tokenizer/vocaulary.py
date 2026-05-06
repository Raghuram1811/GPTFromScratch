"""
Module: vocaulary.py
Author: Raghuram1811
Created: 2026-05-02
Description:
    This module implements vocabulary management for tokenization.
    It provides classes for storing, encoding, and decoding tokens,
    as well as building vocabularies from corpus and special token files.
"""

from uuid import uuid4
import re


class Vocabulary:
    """
    Manages token-to-ID and ID-to-token mappings for tokenization.
    
    This class maintains bidirectional dictionaries to map between tokens (strings)
    and their unique identifiers (UUIDs), allowing for efficient token encoding
    and decoding operations.
    """

    def __init__(self):
        """
        Initialize the Vocabulary with empty mappings.
        
        Attributes:
            token_to_id (dict): Maps tokens (strings) to their unique IDs.
            id_to_token (dict): Maps token IDs to their corresponding token strings.
            additional_tokens (set): Stores additional tokens for reference.
        """
        self.token_to_id = {}
        self.id_to_token = {}
        self.additional_tokens = set()

    def add_token(self, token: str) -> str:
        """
        Add a token to the vocabulary or retrieve its ID if already present.
        
        Args:
            token (str): The token string to add to the vocabulary.
        
        Returns:
            str: The unique ID associated with the token.
        """
        if token in self.token_to_id:
            return self.token_to_id[token]

        token_id = str(uuid4()) 
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        return token_id

    def return_encoded_token_id(self, token: str) -> str:
        """
        Retrieve the ID for a given token, or return unknown token ID if not found.
        
        Args:
            token (str): The token string to encode.
        
        Returns:
            str: The token ID if found, otherwise the ID of '<UNK>' token.
        """
        return self.token_to_id[token] if token in self.token_to_id else self.token_to_id.get('<UNK>')

    def return_decoded_token(self, token_id: str) -> str:
        """
        Retrieve the token string for a given token ID, or return unknown token if not found.
        
        Args:
            token_id (str): The token ID to decode.
        
        Returns:
            str: The token string if found, otherwise the token corresponding to '<UNK>' ID.
        """
        return self.id_to_token[token_id] if token_id in self.id_to_token else self.id_to_token.get(self.token_to_id.get('<UNK>'))


class VocabBuilder:
    """
    Utility class for building vocabularies from corpus and special token files.
    
    This class provides static methods to construct complete vocabularies by
    reading special tokens and corpus files, tokenizing the corpus, and
    populating the vocabulary with all encountered tokens.
    """

    @staticmethod
    def build_vocabulary(corpus_file, special_tokens_file):
        """
        Build a complete vocabulary from a corpus file and special tokens file.
        
        This method reads special tokens from a file first, then reads a corpus,
        tokenizes it using regex pattern splitting, and adds all tokens to the
        vocabulary. Special tokens are prioritized to ensure they are registered
        before corpus tokens.
        
        Args:
            corpus_file (str): Path to the text corpus file to build vocabulary from.
            special_tokens_file (str): Path to the file containing special tokens
                                       (one token per line).
        
        Returns:
            Vocabulary: A populated Vocabulary instance containing all tokens
                       from both special tokens and corpus.
        """
        vocabulary = Vocabulary()

        # Load and add special tokens first
        with open(special_tokens_file, "r") as file:
            special_tokens = [line.strip() for line in file.readlines()]
            for token in special_tokens:
                vocabulary.add_token(token=token)

        # Load corpus and tokenize
        with open(corpus_file, "r") as file:
            text = file.read()
            # Tokenize using regex to split on punctuation, special characters, and whitespace
            tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)

            for token in tokens:
                vocabulary.add_token(token=token)

        return vocabulary
