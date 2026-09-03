"""Sinusoidal time-step embedding for diffusion models — encodes an
integer timestep t into a continuous vector using fixed sin/cos
frequencies, the same construction as `PositionalEncoding`
(utils/layers/embedding.py) but evaluated on the fly for arbitrary
timesteps instead of precomputed sequence positions. Used by
`DenoiserMLP` (models/diffusion/model.py) to condition the denoiser on
which noise level x_t was sampled at.
"""
import numpy as np


def sinusoidal_time_embedding(t, dim):
    """
    Args:
        t: integer timesteps, shape (N,)
        dim: output embedding dimension

    Returns:
        emb: shape (N, dim)
    """
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half) / half)
    args = t[:, None].astype(np.float64) * freqs[None, :]
    emb = np.concatenate([np.sin(args), np.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = np.concatenate([emb, np.zeros((emb.shape[0], 1))], axis=-1)
    return emb
