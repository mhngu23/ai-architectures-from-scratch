"""Training script for `DDPM` (models/diffusion/model.py) on toy 2D data
(data/toy_datasets.py). Run directly with `python -m models.diffusion.train`
from the repo root; see notebooks/week7_demo.ipynb for a visualized,
step-by-step walkthrough of the forward/reverse process on a single
shape (sections 1-5), and training across all 4 toy shapes - joint/shuffled
vs sequential, plus a catastrophic-forgetting check (sections 6-10).
"""
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.diffusion.model import DDPM
from data.toy_datasets import make_two_moons
from utils.loss.MSELoss import MSELoss
from utils.optimizers.Adam import Adam


def fit(model, X, n_steps=4000, batch_size=128, lr=2e-3, log_every=500, verbose=True, optimizer=None):
    """Train `model` (a `DDPM`) with the L_simple objective: sample a
    random timestep t per example, noise x_0 to x_t via `q_sample`, and
    regress the denoiser's noise prediction against the true eps with MSE.

    Every step draws a fresh random minibatch from `X` via `np.random.randint`,
    so training is already implicitly "shuffled" - if `X` is a concatenation
    of several toy shapes, each step's batch is a random mix of all of them.

    Pass an existing `optimizer` (the one returned by a previous `fit` call)
    to keep training the same model on a *different* `X` afterwards while
    preserving Adam's momentum state instead of resetting it - used in
    notebooks/week7_demo.ipynb (sections 6-10) to train sequentially, one
    shape at a time, and check whether the model forgets earlier shapes.

    Returns:
        history: list of per-step losses
        optimizer: the `Adam` instance used (reuse it in a later `fit` call
            to continue training the same model)
    """
    loss_fn = MSELoss()
    if optimizer is None:
        optimizer = Adam(model.parameters(), lr=lr)

    history = []
    for step in range(1, n_steps + 1):
        idx = np.random.randint(0, len(X), size=batch_size)
        x0_batch = X[idx]
        t = np.random.randint(0, model.schedule.timesteps, size=batch_size)

        x_t, eps = model.q_sample(x0_batch, t)
        eps_pred = model(x_t, t)
        loss = loss_fn(eps_pred, eps)

        optimizer.zero_grad()
        model.backward(loss_fn.backward())
        optimizer.step()

        history.append(loss)
        if verbose and (step % log_every == 0 or step == 1):
            print(f"step {step:5d}/{n_steps} | loss = {loss:.4f}")
    return history, optimizer


if __name__ == "__main__":
    np.random.seed(0)
    X = make_two_moons(2000)
    model = DDPM(data_dim=2, time_dim=32, hidden_dim=128, timesteps=200)
    history, _ = fit(model, X, n_steps=4000, batch_size=128, lr=2e-3)

    samples = model.sample(1000)
    print(f"final training loss: {history[-1]:.4f}")
    print(f"real data      mean/std: {X.mean(axis=0)} / {X.std(axis=0)}")
    print(f"generated data mean/std: {samples.mean(axis=0)} / {samples.std(axis=0)}")
