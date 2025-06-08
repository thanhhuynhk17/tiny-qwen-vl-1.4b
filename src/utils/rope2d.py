import torch

def apply_2d_rope(X, cos_theta, sin_theta):
    """
    Applies the 2D Rotary Position Embedding (RoPE) to the input tensor.

    Args:
        X (torch.Tensor): The input tensor to apply RoPE to.
                         Expected shape: (..., sequence_length, embed_dim)
        cos_theta (torch.Tensor): Cosine values for the rotation angles.
                                  Expected shape: (1, sequence_length, embed_dim)
        sin_theta (torch.Tensor): Sine values for the rotation angles.
                                  Expected shape: (1, sequence_length, embed_dim)

    Returns:
        torch.Tensor: The input tensor X after applying the 2D RoPE.
                      Shape will be the same as the input X.

    Notes:
        This function performs the core RoPE rotation operation on the input tensor.
        It assumes X's last dimension is divisible by 2.
    """
    # RoPE 2D
    X_even = X[..., ::2]
    X_odd = X[..., 1::2]
    X_shuffle = torch.stack([-X_odd, X_even], dim=-1).view(X.shape)
    # element-wise multiplication
    X_rope = X * cos_theta + X_shuffle * sin_theta

    return X_rope


def get_rope_2d_angles(embed_dim, grid_size):
    """
    Calculates the cosine and sine values for 2D Rotary Position Embedding (RoPE)
    based on a grid size and embedding dimension.

    Args:
        embed_dim (int): The dimension of the embedding vector. Must be divisible by 4.
        grid_size (int): The size of one side of the square grid (e.g., number of
                         patches along one dimension of an image).

    Returns:
        tuple: A tuple containing two torch.Tensor:
               - cos_theta (torch.Tensor): Cosine values for the rotation angles.
                                         Shape: (1, grid_size*grid_size, embed_dim)
               - sin_theta (torch.Tensor): Sine values for the rotation angles.
                                         Shape: (1, grid_size*grid_size, embed_dim)

    Raises:
        AssertionError: If embed_dim is not divisible by 4.

    Notes:
        This function generates the positional encoding angles for each position
        in a 2D grid by combining 1D encodings for width and height coordinates.
    """
    assert embed_dim % 4 == 0, "Rope 2d requires embed_dim must be divisible by 4"

    # 1) Build a grid of W‐coords and H‐coords, each of shape [grid_size, grid_size].
    coords_w = torch.arange(grid_size, dtype=torch.long)  # [grid_size]
    coords_h = torch.arange(grid_size, dtype=torch.long)  # [grid_size]
    # indexing='xy' makes the first output correspond to x (W), second to y (H)
    grid_w, grid_h = torch.meshgrid(coords_w, coords_h, indexing='xy')
    # flatten to shape [seq_len]
    grid_w = grid_w.flatten()  # [seq_len]
    grid_h = grid_h.flatten()  # [seq_len]

    # 2) Use half of embedding dimensions to encode grid_w, grid_h
    half_dim = embed_dim / 2
    cos_theta_w, sin_theta_w = get_rope_1d_angles(
        half_dim, grid_w)  # [seq_len, embed_dim/4] x2
    cos_theta_h, sin_theta_h = get_rope_1d_angles(
        half_dim, grid_h)  # [seq_len, embed_dim/4] x2

    # [seq_len, embed_dim/2]
    cos_theta = torch.concat([cos_theta_w, cos_theta_h], dim=-1)
    # [seq_len, embed_dim/2]
    sin_theta = torch.concat([sin_theta_w, sin_theta_h], dim=-1)

    # interleave for parallel computing
    cos_theta = cos_theta.repeat_interleave(
        repeats=2, dim=-1)  # [seq_len, embed_dim]
    sin_theta = sin_theta.repeat_interleave(
        repeats=2, dim=-1)  # [seq_len, embed_dim]

    # reshape for broadcasting
    cos_theta = cos_theta.unsqueeze(0)  # [1, seq_len, embed_dim]
    sin_theta = sin_theta.unsqueeze(0)  # [1, seq_len, embed_dim]
    return cos_theta, sin_theta


def get_rope_1d_angles(embed_dim, pos):
    """
    Calculates the cosine and sine values for 1D Rotary Position Embedding (RoPE).

    Args:
        embed_dim (int): The dimension for which to calculate the angles. Must be even.
        pos (torch.Tensor): A tensor containing the positions (indices) for which
                            to calculate the angles. Shape: (..., M)

    Returns:
        tuple: A tuple containing two torch.Tensor:
               - cos_theta (torch.Tensor): Cosine values for the rotation angles.
                                         Shape: (M, embed_dim/2)
               - sin_theta (torch.Tensor): Sine values for the rotation angles.
                                         Shape: (M, embed_dim/2)

    Raises:
        AssertionError: If embed_dim is not even.

    Notes:
        This function generates the base angles for 1D positional encoding.
        It's used as a component in generating 2D RoPE angles.
    """
    assert embed_dim % 2 == 0

    pos = pos.flatten().to(torch.float32)  # [grid_size*grid_size, ]

    i = torch.arange(0, embed_dim / 2)
    inv_freq = 1.0 / (10000 ** (2 * i / embed_dim))  # [embed_dim/2, ]
    # Outer product: [N,] x [freq_dims] → [N, freq_dims]
    # [grid_size*grid_size, embed_dim/2]
    angles = pos.unsqueeze(1) @ inv_freq.unsqueeze(0)

    # each [grid_size*grid_size, embed_dim/2]
    return torch.cos(angles), torch.sin(angles)
