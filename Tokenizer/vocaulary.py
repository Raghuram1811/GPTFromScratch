from uuid import uuid4
import re


class Vocabulary:

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.additional_tokens = set()

    def add_token(self, token: str) -> str:
        if token in self.token_to_id:
            return self.token_to_id[token]

        token_id = str(uuid4()) 
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        return token_id

    def return_encoded_token_id(self, token: str) -> str:
        return self.token_to_id[token] if token in self.token_to_id else self.token_to_id.get('<UNK>')

    def return_decoded_token(self, token_id: str) -> str:
        return self.id_to_token[token_id] if token_id in self.id_to_token else self.id_to_token.get(self.token_to_id.get('<UNK>'))


class VocabBuilder:

    @staticmethod
    def build_vocabulary(corpus_file, special_tokens_file): 
        vocabulary = Vocabulary()

        with open(special_tokens_file, "r") as file:
            special_tokens = [line.strip() for line in file.readlines()]
            for token in special_tokens:
                vocabulary.add_token(token=token)

        with open(corpus_file, "r") as file:
            text = file.read()
            tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)

            for token in tokens:
                vocabulary.add_token(token=token)

        return vocabulary