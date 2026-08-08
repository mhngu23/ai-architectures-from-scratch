"""Binary Cross-Entropy Loss — hàm loss dùng để huấn luyện
`TabularTransformer`/`TextClassifierTransformer` (và MLP) cho bài toán
phân loại nhị phân: so sánh xác suất dự đoán (đầu ra của Sigmoid) với
nhãn 0/1 thật.
"""
import numpy as np

class BCELoss:
    def __call__(self, pred, target):
        self.pred = pred
        self.target = target
        eps = 1e-8
        return -np.mean(target * np.log(pred + eps) + (1 - target) * np.log(1 - pred + eps))

    def backward(self):
        eps = 1e-8
        return (self.pred - self.target) / ((self.pred + eps) * (1 - self.pred + eps)) / self.pred.size
