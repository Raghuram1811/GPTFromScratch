
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

        pair_frequency = sorted(self.rank_pairs(b_array).items(), key = lambda x: x[1], reverse = True)
        print(pair_frequency)

        # Testing the merge logic for the byte-pairs
        merged = self.merge_tokens(256, (116, 104), b_array)
        
        print(merged)
            

def main():
    bpe_tokenizer = BPETokenizer(vocab=dict())
    bpe_tokenizer.tokenize(input="the cat in the hat")
    #"ant ant ant ant ant ant ants ants ants plant plant plants gigantic gigantic gigantic")


if __name__ == '__main__':
    main()