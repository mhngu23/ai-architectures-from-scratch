"""A few plain NumPy functions (not layer classes) for ReLU/MSE — the
original simple implementation from Week 1, only still used in
`tests/test_modules.py`. Not directly related to the Transformer; the
versions actually used by the models are the classes in
`utils/activations/` and `utils/loss/`.
"""
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def mse_loss(pred, target):
    return 0.5 * ((pred - target) ** 2).mean()

def mse_loss_grad(pred, target):
    return (pred - target) / pred.size
