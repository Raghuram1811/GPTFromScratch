
from vocaulary import VocabBuilder

import os
import sys

def main(sample_token=None):

    corpus_file = os.path.join(os.path.abspath("Corpus"), "sample.txt")
    special_tokens = os.path.join(os.path.abspath("Corpus"), "special_tokens.txt")

    # Build the vocabulary from the corpus file
    vocabBuilder = VocabBuilder()
    vocabulary = vocabBuilder.build_vocabulary(corpus_file=corpus_file, special_tokens_file=special_tokens)
    print("Vocabulary built successfully. Sample tokens and their ids:")
    for token, token_id in vocabulary.token_to_id.items():
        print(f"'{token}': {token_id}")

    # Encode the sample from corupus file
    encoded_token_id = vocabulary.return_encoded_token_id(sample_token)
    print(f"Encoded token for '{sample_token}': {encoded_token_id}")

    #Decode the token id back to token
    token_id = vocabulary.token_to_id.get(sample_token)
    decoded_token = vocabulary.return_decoded_token(token_id)
    print(f"Decoded token for id '{token_id}': {decoded_token}")

if __name__ == "__main__":
    main(sample_token=sys.argv[1] if len(sys.argv) >= 2 else "sample")