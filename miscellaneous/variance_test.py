import numpy as np

if __name__ == "__main__":
    print("This is a test file for variance testing.")

    # Generate a small random dataset
    data = np.random.random_integers(low=0, high=100, size=(10))
    print("Data:", data) 

    # Applying softmax to the scaled data without normalization
    exp_data = np.exp(data)
    softmax_data = exp_data / np.sum(exp_data)
    print("Softmax Data without normalization:", softmax_data)
    print("sum of softmax data after normalization:", np.sum(softmax_data))

    # Divide the data with the sqrt of the dimension (which is 10 in this case) - Applying normalization to prevent overflow in softmax
    dimension = len(data)
    scaled_data = data / np.sqrt(dimension)
    print("Scaled Data:", scaled_data)

    # Applying softmax to the scaled data
    exp_data = np.exp(scaled_data)
    softmax_data = exp_data / np.sum(exp_data)
    print("Softmax Data after normalization:", softmax_data)

    print("sum of softmax data after normalization:", np.sum(softmax_data))