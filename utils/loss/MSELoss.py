"""Mean Squared Error Loss — dùng cho bài toán hồi quy/tái tạo (vd
Autoencoder tái tạo lại input của chính nó). Không được dùng trực tiếp bởi
Transformer trong repo này (Transformer dùng `BCELoss` cho phân loại nhị
phân), có sẵn cho các model khác cần loss dạng hồi quy.
"""
class MSELoss:
    def __call__(self, pred, target):
        self.pred = pred
        self.target = target
        return ((pred - target) ** 2).mean()

    def backward(self):
        return 2 * (self.pred - self.target) / self.pred.size
