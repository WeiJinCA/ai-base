import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(max_len, d_model):
    """
    Generate positional encoding for a sequence of max_len and embedding size d_model.

    Args:
        max_len (int): Maximum sequence length.
        d_model (int): Embedding dimension.

    Returns:
        numpy.ndarray: Positional encoding matrix of shape (max_len, d_model).
    """
    # Initialize the encoding matrix with zeros
    pe = np.zeros((max_len, d_model))

    # Create a position vector (0, 1, 2, ..., max_len-1)
    position = np.arange(max_len).reshape(-1, 1)  # Shape: (max_len, 1)

    # Compute the denominator term: 10000^(2i/d_model)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    # Apply sine to even indices (2i) and cosine to odd indices (2i+1)
    pe[:, 0::2] = np.sin(position * div_term)  # Even indices
    pe[:, 1::2] = np.cos(position * div_term)  # Odd indices

    return pe

# Example Usage
max_len = 100   # Sequence length
d_model = 16    # Embedding dimension

pos_enc = positional_encoding(max_len, d_model)

# Visualizing Positional Encoding
plt.figure(figsize=(10, 6))
plt.imshow(pos_enc, cmap='viridis', aspect='auto')
plt.colorbar(label="Encoding Value")
plt.xlabel("Embedding Dimension")
plt.ylabel("Position in Sequence")
plt.title("Positional Encoding Heatmap")
plt.show()