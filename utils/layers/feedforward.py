from utils.layers.linear import Linear
from utils.activations.ReLU import ReLU


class PositionwiseFeedForward:
    """The Transformer's per-position FFN: Linear -> ReLU -> Linear.

    Applied independently at every sequence position, so (batch, seq_len,
    d_model) is flattened to (batch * seq_len, d_model) to reuse the 2D
    `Linear` layer, then restored to its original shape.
    """
    def __init__(self, d_model, d_ff):
        self.linear1 = Linear(d_model, d_ff, activation='relu')
        self.relu = ReLU()
        self.linear2 = Linear(d_ff, d_model)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        h = self.linear1(x.reshape(batch * seq_len, d_model))
        h = self.relu(h)
        out = self.linear2(h)
        return out.reshape(batch, seq_len, d_model)

    def backward(self, grad_output):
        batch, seq_len, d_model = grad_output.shape
        grad = self.linear2.backward(grad_output.reshape(batch * seq_len, d_model))
        grad = self.relu.backward(grad)
        grad = self.linear1.backward(grad)
        return grad.reshape(batch, seq_len, d_model)

    def parameters(self):
        return [self.linear1.W, self.linear1.b, self.linear2.W, self.linear2.b]
