"""Numerical gradient checks for the encoder/decoder building blocks.

Every module here implements backward() by hand (no autograd), so the only
real way to trust it is to compare against finite-difference gradients on a
scalar loss. Shapes are kept tiny (batch=2, seq=3, d_model=4) purely to keep
the O(n) finite-difference sweep fast.
"""
import numpy as np

from utils.layers.layernorm import LayerNorm
from utils.layers.attention import ScaledDotProductAttention, MultiHeadAttention
from utils.layers.feedforward import PositionwiseFeedForward
from models.transformer.model import EncoderLayer, DecoderLayer, FeatureTokenizer, TabularTransformer
from models.autoencoder.model import Autoencoder

np.random.seed(0)
EPS = 1e-4
TOL = 1e-3


def numerical_grad(f, x, eps=EPS):
    """Central-difference gradient of scalar-valued f w.r.t. every entry of x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        plus = f()
        x[idx] = orig - eps
        minus = f()
        x[idx] = orig
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


def relative_error(a, b):
    return np.max(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-8))


def check_grad(name, analytic, numeric):
    err = relative_error(analytic, numeric)
    status = "OK" if err < TOL else "FAIL"
    print(f"  [{status}] {name}: max relative error = {err:.2e}")
    assert err < TOL, f"{name} gradient check failed (err={err:.2e})"


def loss_and_grad(out):
    """L = 0.5 * sum(out^2)  =>  dL/dout = out."""
    return 0.5 * np.sum(out ** 2), out


def check_module(module, inputs, input_names, param_names=None):
    """Runs forward, backward, and numerically checks gradients for every
    array in `inputs` plus every Parameter returned by module.parameters()
    (if `param_names` is given, matching length/order to parameters()).
    """
    def forward_all():
        out = module.forward(*inputs)
        loss, _ = loss_and_grad(out)
        return loss

    out = module.forward(*inputs)
    _, grad_out = loss_and_grad(out)
    result = module.backward(grad_out)
    analytic_grads = result if isinstance(result, tuple) else (result,)

    for name, x, analytic in zip(input_names, inputs, analytic_grads):
        numeric = numerical_grad(forward_all, x)
        check_grad(name, analytic, numeric)

    if param_names is not None:
        for name, p in zip(param_names, module.parameters()):
            def forward_all_reset():
                return forward_all()
            numeric = numerical_grad(forward_all_reset, p.data)
            check_grad(name, p.grad, numeric)


def test_layernorm():
    print("LayerNorm")
    np.random.seed(0)
    ln = LayerNorm(4)
    x = np.random.randn(2, 3, 4)
    check_module(ln, [x], ["x"], [f"param[{i}]" for i in range(2)])


def test_scaled_dot_product_attention():
    print("ScaledDotProductAttention")
    np.random.seed(0)
    attn = ScaledDotProductAttention()
    Q = np.random.randn(2, 2, 3, 4)
    K = np.random.randn(2, 2, 3, 4)
    V = np.random.randn(2, 2, 3, 4)
    check_module(attn, [Q, K, V], ["Q", "K", "V"])


def test_multihead_attention():
    print("MultiHeadAttention")
    np.random.seed(0)
    mha = MultiHeadAttention(d_model=4, num_heads=2)
    Q = np.random.randn(2, 3, 4)
    K = np.random.randn(2, 3, 4)
    V = np.random.randn(2, 3, 4)
    check_module(mha, [Q, K, V], ["Q_in", "K_in", "V_in"], [f"param[{i}]" for i in range(8)])


def test_feedforward():
    print("PositionwiseFeedForward")
    np.random.seed(0)
    ff = PositionwiseFeedForward(d_model=4, d_ff=6)
    x = np.random.randn(2, 3, 4)
    check_module(ff, [x], ["x"], [f"param[{i}]" for i in range(4)])


def test_encoder_layer():
    print("EncoderLayer")
    np.random.seed(0)
    layer = EncoderLayer(d_model=4, num_heads=2, d_ff=6)
    x = np.random.randn(2, 3, 4)
    n_params = len(layer.parameters())
    check_module(layer, [x], ["x"], [f"param[{i}]" for i in range(n_params)])


def test_decoder_layer():
    print("DecoderLayer")
    np.random.seed(0)
    layer = DecoderLayer(d_model=4, num_heads=2, d_ff=6)
    x = np.random.randn(2, 1, 4)
    enc_out = np.random.randn(2, 3, 4)

    def forward_all():
        out = layer.forward(x, enc_out)
        loss, _ = loss_and_grad(out)
        return loss

    out = layer.forward(x, enc_out)
    _, grad_out = loss_and_grad(out)
    grad_x, grad_enc = layer.backward(grad_out)

    check_grad("x", grad_x, numerical_grad(forward_all, x))
    check_grad("enc_out", grad_enc, numerical_grad(forward_all, enc_out))
    for i, p in enumerate(layer.parameters()):
        check_grad(f"param[{i}]", p.grad, numerical_grad(forward_all, p.data))


def test_feature_tokenizer():
    print("FeatureTokenizer")
    np.random.seed(0)
    tok = FeatureTokenizer(num_features=3, d_model=4)
    x = np.random.randn(2, 3)
    check_module(tok, [x], ["x"], [f"param[{i}]" for i in range(6)])


def test_tabular_transformer():
    print("TabularTransformer (full encoder-decoder model)")
    np.random.seed(0)
    model = TabularTransformer(num_features=3, d_model=4, num_heads=2, d_ff=6, num_layers=1)
    x = np.random.randn(2, 3)
    n_params = len(model.parameters())
    check_module(model, [x], ["x"], [f"param[{i}]" for i in range(n_params)])


def test_autoencoder():
    print("Autoencoder")
    np.random.seed(0)
    model = Autoencoder(input_dim=5, hidden_dims=[6], latent_dim=3)
    x = np.random.randn(2, 5)
    n_params = len(model.parameters())
    check_module(model, [x], ["x"], [f"param[{i}]" for i in range(n_params)])


if __name__ == "__main__":
    for fn in [
        test_layernorm,
        test_scaled_dot_product_attention,
        test_multihead_attention,
        test_feedforward,
        test_encoder_layer,
        test_decoder_layer,
        test_feature_tokenizer,
        test_tabular_transformer,
        test_autoencoder,
    ]:
        fn()
    print("\nAll gradient checks passed.")
