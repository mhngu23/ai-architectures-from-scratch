"""Minimal tests for the plain NumPy functions in `utils/common.py` — not
directly related to the Transformer.
"""
import numpy as np
from utils.common import relu

def test_linear_forward(layer=None, x=None):
    out = layer(x)
    assert out.shape == (1, 2)
    print("Output:", out)
    print("✅ test_linear_forward passed")

def test_relu(x):
    out = relu(x)
    print("Output:", out)
    assert (out == np.array([[0, 0, 2]])).all()
    print("✅ test_relu passed")
