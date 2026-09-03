"""Toy 2D datasets for the diffusion model (models/diffusion/). Each
function returns an (N, 2) array normalized to mean ~0 / std ~1, so every
shape is a comparable match for the N(0, I) prior `DDPM.sample` starts
from. Used directly in models/diffusion/train.py, and combined/compared
in notebooks/week7_demo.ipynb (sections 6-10: multi-shape training +
catastrophic-forgetting check).
"""
import numpy as np


def make_two_moons(n_samples=2000, noise=0.05, seed=0):
    """2 interleaving half-circles."""
    rng = np.random.default_rng(seed)
    n1 = n_samples // 2
    n2 = n_samples - n1

    theta1 = np.linspace(0, np.pi, n1)
    x1 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1)

    theta2 = np.linspace(0, np.pi, n2)
    x2 = np.stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5], axis=1)

    X = np.concatenate([x1, x2], axis=0)
    X += rng.normal(scale=noise, size=X.shape)

    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X.astype(np.float64)


def make_swiss_roll_2d(n_samples=2000, noise=0.05, seed=0):
    """A 2D (flattened) Swiss roll spiral."""
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(size=n_samples))
    x = t * np.cos(t)
    y = t * np.sin(t)
    X = np.stack([x, y], axis=1)
    X += rng.normal(scale=noise, size=X.shape)

    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X.astype(np.float64)


def make_gaussian_mixture(n_samples=2000, n_components=8, radius=3.0, std=0.15, seed=0):
    """`n_components` Gaussian blobs arranged evenly around a ring."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    comp_idx = rng.integers(0, n_components, size=n_samples)
    X = centers[comp_idx] + rng.normal(scale=std, size=(n_samples, 2))

    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X.astype(np.float64)


def make_checkerboard(n_samples=2000, n_cells=4, seed=0):
    """Points filling the "black" squares of an `n_cells` x `n_cells`
    checkerboard - a sharply multi-modal shape (several disconnected
    square blobs), useful as a visually distinct 4th shape when testing
    whether the model still covers earlier shapes after training on this
    one (see notebooks/week7_demo.ipynb, sections 6-10)."""
    rng = np.random.default_rng(seed)
    samples = []
    while len(samples) < n_samples:
        batch = rng.uniform(-n_cells / 2, n_cells / 2, size=(n_samples, 2))
        cell = np.floor(batch).astype(int)
        keep = (cell[:, 0] + cell[:, 1]) % 2 == 0
        samples.append(batch[keep])
    X = np.concatenate(samples, axis=0)[:n_samples]

    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X.astype(np.float64)


TOY_DATASETS = {
    "two_moons": make_two_moons,
    "swiss_roll": make_swiss_roll_2d,
    "gaussian_mixture": make_gaussian_mixture,
    "checkerboard": make_checkerboard,
}
