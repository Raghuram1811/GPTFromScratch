
import torch
import os
import sys

# Setup project root path for module imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessor.data.prepare_data import DataPreProcessor, TorchWrapper

def main():

    text = """ Sachin Ramesh Tendulkar is a former Indian international cricketer and a former captain of the Indian national team. He is widely regarded as one of the greatest batsmen in the history of cricket. Tendulkar is the highest run scorer of all time in international cricket, with more than 34,000 runs in both Test and One Day International (ODI) formats. He is also the only player to have scored one hundred international centuries, which includes 51 Test centuries and 49 ODI centuries. Tendulkar made his debut for India at the age of 16 and had a career that spanned over two decades, during which he set numerous records and achieved many milestones in the sport. He retired from international cricket in 2013 and has since been involved in various philanthropic activities and cricket-related initiatives."""
    
    # Create dataset with small context window and stride for demonstration
    dataset = DataPreProcessor(text, model="gpt2", context_length=5, stride=5)
    # Wrap dataset for batch iteration
    torch_wrapper = TorchWrapper(dataset=dataset, batch_size=20, shuffle=False)

    embedding_layer = torch.nn.Embedding(num_embeddings=50257, embedding_dim=5) # Create an embedding layer with 10 tokens and dimension of 50257 (GPT-2 vocab size)

    print(embedding_layer.weight) # Should return the randomly initialized embedding weights of shape (10, 5)
    for batch in torch_wrapper.dataloader:
        print(type(batch), batch.shape) # Should return the type and shape of the batch, e.g. <class 'torch.Tensor'> torch.Size([20, 4]) where 20 is the batch size and 4 is the context length
        embedded_batch = embedding_layer(batch) # Pass the batch through the embedding layer
        print(embedded_batch.shape) # Should return the shape of the embedded batch, e.g. torch.Size([20, 4, 5]) where 5 is the embedding dimension


if __name__ == '__main__':
    main()