import os
import sys
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessor.Tokenizer.tiktokenizer import TikTokenizer
from torch.utils.data import DataLoader, Dataset

class DataPreProcessor(Dataset):

    def __init__(self, text, model, context_length, stride):
        self.input = []
        self.target = []

        tiktokenizer = TikTokenizer(model = model)
        token_ids = tiktokenizer.encode(input = text, allowed_special={"<|endoftext|>"})
        
        for i in range(0, len(token_ids)-context_length, stride):
            input_row = token_ids[i:i+context_length]
            target_row = token_ids[i+1:i+context_length+1]

            # print(input_row, target_row)

            self.input.append(torch.tensor(input_row))
            self.target.append(torch.tensor(target_row))
        # print(f"Total input sequences: {self.input}")
        # print(f"Total target sequences: {self.target}")
    
    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]
    
    def __gettensors__(self, idx):
        return torch.tensor(self.input[idx]), torch.tensor(self.target[idx])

class TorchWrapper:

    """ Wrapper class for PyTorch DataLoader to handle batching and shuffling of the dataset. """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=2)
        

def main():
    text = """Tendulkar took up cricket at the age of eleven, made his Test match debut on 15 November 1989 against Pakistan in Karachi at the age of sixteen, and went on to represent Mumbai domestically and India internationally for over 24 years.[11] In 2002, halfway through his career, Wisden ranked him the second-greatest Test batsman of all time, behind Don Bradman, and the second-greatest ODI batsman of all time, behind Viv Richards.[12] The same year, Tendulkar was a part of the team that was one of the joint-winners of the 2002 ICC Champions Trophy. Later in his career, Tendulkar was part of the Indian team that won the 2011 Cricket World Cup, his first win in six World Cup appearances for India.[13] He had previously been named "Player of the Tournament" at the 2003 World Cup.

Tendulkar has received several awards from the government of India: the Arjuna Award (1994), the Khel Ratna Award (1997), the Padma Shri (1998), and the Padma Vibhushan (2008).[14][15] After Tendulkar played his last match in November 2013, the Prime Minister's Office announced the decision to award him the Bharat Ratna, India's highest civilian award.[16][17] He was the first sportsperson to receive the award and, as of 2024, is the youngest recipient.[18][19][20] Having retired from ODI cricket in 2012,[21][22] he retired from all forms of cricket in November 2013 after playing his 200th Test match.[23] Tendulkar played 664 international cricket matches in total, scoring 34,357 runs.[24] In 2013, Tendulkar was included in an all-time Test World XI to mark the 150th anniversary of Wisden Cricketers' Almanack, and he was one of only two specialist batsmen of the post–World War II era, along with Viv Richards, to get featured in the team.[25]

Tendulkar is regarded as a symbol of national pride in India for his achievements. In 2010, Time included Tendulkar in its annual list of the most influential people in the world.[26] Tendulkar was awarded the Sir Garfield Sobers Trophy for cricketer of the year at the 2010 International Cricket Council (ICC) Awards.[27] In 2019, he was inducted into the ICC Cricket Hall of Fame.[28]"""
    
    dataset = DataPreProcessor(text, model="gpt2", context_length=4, stride = 4)
    torch_wrapper = TorchWrapper(dataset=dataset, batch_size=20, shuffle=False)
    # print("DataLoader created successfully. Sample batches:")
    # for idx, batch in enumerate(torch_wrapper.dataloader):
    #     print(idx+1, batch)

if __name__ == '__main__':
    main()