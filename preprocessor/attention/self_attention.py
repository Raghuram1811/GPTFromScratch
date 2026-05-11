import torch
import os
import sys

# Setup project root path for module imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessor.data.prepare_data import DataPreProcessor, TorchWrapper



def main():
    text = """ The next day is brighter since we started learning about self attention. We can now understand how each word in a sentence can influence the representation of other words, allowing us to capture long-range dependencies and contextual relationships more effectively. This is a crucial step in building powerful language models like GPT, which rely on self attention mechanisms to generate coherent and contextually relevant text. With this knowledge, we can now explore how to implement self attention in our own models and further enhance our understanding of natural language processing. """    

    # Step 1: Prepare the dataset using DataPreProcessor
    dataset = DataPreProcessor(text=text, model="gpt2", context_length=5, stride=2)
    torch_wrapper = TorchWrapper(dataset=dataset, batch_size=1, shuffle=False)
    print(f"Dataset created with {len(dataset)} samples")
    
    # Step 2: Attention mechanism demonstration
    input_dimension = 5 # For the query, key, value vectors - 5 is just an example dimension for demonstration purposes. In a real model, this would typically be the embedding dimension (e.g., 768 for GPT-2 small).  
    output_dimension = 5 # Output dimension of the attention layer - 5, In a real model, this would typically be the embedding dimension (e.g., 768 for GPT-2 small).  

    W_q = W_k = W_v = torch.nn.Parameter(torch.randn(input_dimension, output_dimension), requires_grad=False)  # Query, key and value weight matrices initialized.
    print(f"\n\n Initialized weight matrices W_q, W_k, W_v with shape: {W_q.shape}, {W_k.shape}, {W_v.shape} \n\n")

    # Now to get the query vector for a sample input row from dataset above, we would do:
    sample_input = torch_wrapper.dataloader.dataset[1]  # Get the input row for the first sample.
    print(f"Sample input from dataset: {sample_input}\n")

    query_vector_sample = sample_input.float() @ W_q   # Compute the query vector by multiplying the input with the query weight matrix.
    print(f"For input sample {sample_input}, the computed query vector has shape: {query_vector_sample.shape} and values: \n{query_vector_sample}\n \n")

    # Similarly, we can compute the key and value vectors for the same input sample:
    key_vector_sample = sample_input.float() @ W_k     # Compute the key vector by multiplying the input with the key weight matrix.  
    value_vector_sample = sample_input.float() @ W_v   # Compute the value vector by multiplying the input with the value weight matrix.
    print(f"For input sample {sample_input}, the computed key vector has shape: {key_vector_sample.shape} and values: \n{key_vector_sample}\n \n")
    print(f"For input sample {sample_input}, the computed value vector has shape: {value_vector_sample.shape} and values: \n{value_vector_sample}\n \n")

    # Now that we got the Query, Key matrices, we need to compute Q.K^T to get the attention scores, and then apply softmax to get the attention weights. 
    # Finally, we will multiply the attention weights with the value vector to get the output of the attention mechanism for this sample input.

    print("==** Attention Score Calculation (Q.K^T) ***==")
    # Let us check the shape of Q and K^T before we do the multiplication to ensure they are compatible for matrix multiplication.
    print(f"\n\nShape of query vector: {query_vector_sample.shape}", f"Shape of key^T vector: {key_vector_sample.T.shape}")
    attention_scores = q_k_transpose = query_vector_sample @ key_vector_sample.T  # Compute Q.K^T to get attention scores.
    print(f"Shape of Q.K^T (attention scores): {q_k_transpose.shape} and values: \n{q_k_transpose}\n \n")
    attention_scores = attention_scores / (output_dimension ** 0.5)  # Scale the attention scores by the square root of the output dimension (d_k) to prevent large values that can lead to vanishing gradients.
    print(f"Attention score normalized and soft-maxed: {attention_scores} \n \n")
    print("==** Attention Output Calculation (Attention Weights . Value) ***==")

    attention_output = attention_scores * value_vector_sample  # Multiply the attention weights with the value vector to get the output of the attention mechanism.
    print(f"Shape of attention output: {attention_output.shape} and values: \n{attention_output}\n \n")

if __name__ == "__main__":
    print("This is a test file for implementing self attention mechanism.")
    main()