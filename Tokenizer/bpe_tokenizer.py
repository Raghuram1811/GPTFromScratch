import math


class BPETokenizer():

    def __init__(self):
        self.merges = dict()
        self.next_id = 256
        self.vocab = {i: bytes([i]) for i in range(0, 256)}
    
    def rank_pairs(self, token_ids: list):
        counts={}
        for idx in range(len(token_ids)-1):
            pair = (token_ids[idx], token_ids[idx+1])
            counts[pair] = counts.get(pair, 0)+1
        return counts
    
    def merge_tokens(self, new_id, pair, token_ids):

        new_tokens, idx = [], 0
        while idx < len(token_ids):
            if idx < len(token_ids)-1 and pair == (token_ids[idx], token_ids[idx+1]):
                new_tokens.append(new_id)
                idx+=2
            else:   
                new_tokens.append(token_ids[idx])
                idx+=1
        return new_tokens

    def tokenize(self, b_array: list[int], top_frequency:float = math.inf):

        if top_frequency <= 1:
            return b_array

        pair_counts = self.rank_pairs(b_array)

        if not pair_counts:
            return b_array

        top_pair, top_frequency = sorted(pair_counts.items(), key = lambda x: x[1], reverse = True)[0]

        id = self.next_id
        self.next_id+=1

        self.merges[top_pair] = id
        self.vocab[id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

        b_array = self.merge_tokens(id, top_pair, b_array)

        return self.tokenize(b_array=b_array, top_frequency=top_frequency)

def main():
    
    input_string = """ant ant ant ant ant ant ants ants ants plant plant plants gigantic gigantic gigantic"""
    encoded_string = input_string.encode('utf-8')
    
    bpe_tokenizer = BPETokenizer()
    bpe_tokenizer.tokenize(b_array=list(encoded_string))

    # for pair, new_id in bpe_tokenizer.merge.items():
    #     print(pair, "=>", new_id, bpe_tokenizer.vocab[new_id])
    
    # Print the updated Vocabulary after BPE-Tokenization is done.
    for item in bpe_tokenizer.vocab.items():
        print(item)


if __name__ == '__main__':
    main()