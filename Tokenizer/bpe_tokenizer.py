import random


class BPETokenizer():

    def __init__(self, vocab: dict):
        self.vocab = vocab

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

    def tokenize(self, input: str):

        b_array = list(input.encode('utf-8'))
        
        for i in range(100):

            print(b_array)

            top_pair, top_frequency = sorted(self.rank_pairs(b_array).items(), key = lambda x: x[1], reverse = True)[0]

            print(top_pair, top_frequency)
            
            if top_frequency ==1:
                break
            id = random.sample(range(256, 500), 1).pop()

            # Testing the merge logic for the byte-pairs
            b_array = self.merge_tokens(id, top_pair, b_array)
            

def main():
    bpe_tokenizer = BPETokenizer(vocab=dict())
    bpe_tokenizer.tokenize(input="ant ant ant ant ant ant ants ants ants plant plant plants gigantic gigantic gigantic")
    #"ant ant ant ant ant ant ants ants ants plant plant plants gigantic gigantic gigantic")


if __name__ == '__main__':
    main()