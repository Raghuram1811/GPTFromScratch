import tiktoken

class TikTokenizer():

    def __init__(self, model: str):
        self.model = model
    

    def encoder(self, input: str) -> str:
        return  tiktoken.get_encoding(self.model).encode(input)
    

    def decode(self, input: str) -> str:
        return tiktoken.get_encoding(self.model).decode(input)

    
# Testing tiktoken library with gpt2
def main():
    tikTokenizer = TikTokenizer('gpt2')
    print(tikTokenizer.decode(tikTokenizer.encoder("The cat sat on the mat")))   # Should return the input string sent, i.e "The cat sat on the mat"

if __name__ == '__main__':
    main()