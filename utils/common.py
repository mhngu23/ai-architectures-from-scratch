"""Vài hàm NumPy thuần (không phải class layer) cho ReLU/MSE — bản viết
đơn giản ban đầu từ Week 1, chỉ còn dùng trong `tests/test_modules.py`.
Không liên quan trực tiếp đến Transformer; các phiên bản dùng thật trong
model là các class ở `utils/activations/` và `utils/loss/`.
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
