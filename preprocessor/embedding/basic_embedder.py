
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
    dataset = DataPreProcessor(text, model="gpt2", context_length=5, stride=2)
    # Wrap dataset for batch iteration
    torch_wrapper = TorchWrapper(dataset=dataset, batch_size=20, shuffle=False)

    """ 
        Embedding Matrix is like a lookup table that maps each token ID to a dense vector representation (embedding). Created raw at 1st iteration using Embedding class in PyTorch. 
        
        The num_embeddings parameter specifies the size of the vocabulary (number of unique tokens), and embedding_dim specifies the dimensionality of the embedding vectors. In this example, we are creating an embedding layer with 50257 tokens (the size of GPT-2's vocabulary) and an embedding dimension of 5 for demonstration purposes. In practice, you would typically use a larger embedding dimension (e.g., 768 or 1024) for better performance, but we are using a smaller dimension here to keep it simple and easy to understand. The weights of the embedding layer are randomly initialized when the layer is created, and they will be updated during training as the model learns to represent the tokens in a way that captures their semantic meaning and relationships 
        based on the training data. The embedding layer will learn to assign similar vectors to tokens that appear in similar contexts, allowing the model to capture semantic relationships between words and improve its ability to understand and generate text.
    
        key differenece between vocabulary size and embedding dimension is that vocabulary size refers to the total number of unique tokens in the dataset, while embedding dimension refers to the size of the vector representation for each token. The embedding layer maps each token ID (from the vocabulary) to a dense vector of a specified dimension, allowing the model to learn meaningful representations of the tokens based on their usage in the training data.

        So for each cell in input tensor of shape (batch_size, context_length) containing token IDs, the embedding layer will output a corresponding tensor of shape (batch_size, context_length, embedding_dim) where each token ID is replaced by its learned embedding vector.
    
        Now, how the input tensor shape and Embedding Matrix shape interact is that the input tensor contains token IDs that correspond to the indices of the embedding matrix. When you pass the input tensor through the embedding layer, it looks up the corresponding embedding vector for each token ID in the input tensor and outputs a new tensor where each token ID is replaced by its embedding vector. For example, if your input tensor has a shape of (batch_size, context_length) and contains token IDs, and your embedding layer has a shape of (num_embeddings, embedding_dim), then the output of the embedding layer will have a shape of (batch_size, context_length, embedding_dim), where each token ID in the input tensor is replaced by its corresponding embedding vector from the embedding matrix.
        This allows the model to capture semantic relationships between tokens and improve its ability to understand and generate text, as the embedding vectors can encode information about the meaning and context of the tokens based on their usage in the training data.
    """
    num_embeddings = 50257 # Size of GPT-2's vocabulary
    embedding_dim = 768 # For demonstration
    context_length = 5 # For demonstration


    embedding_layer = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim) # Create an embedding layer with 10 tokens and dimension of 50257 (GPT-2 vocab size)

    positional_embeddin_layer = torch.nn.Embedding(num_embeddings=context_length, embedding_dim=embedding_dim) # Create a positional embedding layer with max position of 5 and dimension of 768 (same as token embedding dimension)

    positions = torch.arange(context_length) # context length is 5, so we create a tensor of positions from 0 to 4
    positional_embeddings = positional_embeddin_layer(positions) # Get the positional embeddings for the positions tensor
    for batch in torch_wrapper.dataloader:
        token_embeddings = embedding_layer(batch) # Pass the batch through the token embedding layer to get token embeddings of shape (batch_size, context_length, embedding_dim)   
        print(type(batch), batch.shape) # Should return the type and shape of the batch, e.g. <class 'torch.Tensor'> torch.Size([20, 4]) where 20 is the batch size and 4 is the context length
        embedded_batch = token_embeddings + positional_embeddings # Pass the batch through the embedding layer
        print(embedded_batch) # Should return the shape of the embedded batch, e.g. torch.Size([20, 4, 5]) where 5 is the embedding dimension
        print(embedded_batch.shape) # Should return the shape of the embedded batch, e.g. torch.Size([20, 4, 5]) where 5 is the embedding dimension
if __name__ == '__main__':
    main()
