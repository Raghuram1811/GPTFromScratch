from uuid import uuid4
import re


class Vocabulary:

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

    def add_token(self, token: str) -> str:
        if token in self.token_to_id:
            return self.token_to_id[token]

        token_id = str(uuid4()) 
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        return token_id

    def return_encoded_token_id(self, token: str) -> str:
        return self.token_to_id[token]

    def return_decoded_token(self, token_id: str) -> str:
        return self.id_to_token[token_id]


class VocabBuilder:

    @staticmethod
    def build_vocabulary(corpus_file): 
        vocabulary = Vocabulary()

        with open(corpus_file, "r") as file:
            text = file.read()
            tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)

            for token in tokens:
                vocabulary.add_token(token=token)

        return vocabulary