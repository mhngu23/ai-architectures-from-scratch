import numpy as np
from utils.common import Linear, relu, relu_derivative, mse_loss, mse_loss_grad

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

