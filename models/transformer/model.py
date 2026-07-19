import numpy as np

from utils.layers.linear import Linear, Parameter
from utils.layers.layernorm import LayerNorm
from utils.layers.attention import MultiHeadAttention
from utils.layers.feedforward import PositionwiseFeedForward
from utils.activations.Sigmoid import Sigmoid


class EncoderLayer:
    """Self-attention + FFN, each wrapped in a residual connection and a
    post-norm (`LayerNorm(x + Sublayer(x))`), matching "Attention Is All You
    Need" rather than the pre-norm variant used by later models.
    """
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)

    def __call__(self, x, mask=None):
        return self.forward(x, mask)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

    def backward(self, grad_output):
        grad_sum2 = self.norm2.backward(grad_output)
        grad_ff_in = self.ff.backward(grad_sum2)
        grad_x1 = grad_sum2 + grad_ff_in

        grad_sum1 = self.norm1.backward(grad_x1)
        grad_q, grad_k, grad_v = self.self_attn.backward(grad_sum1)
        grad_x = grad_sum1 + grad_q + grad_k + grad_v
        return grad_x

    def parameters(self):
        return (self.self_attn.parameters() + self.norm1.parameters()
                + self.ff.parameters() + self.norm2.parameters())


class Encoder:
    """A stack of `num_layers` EncoderLayers."""
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        self.layers = [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def __call__(self, x, mask=None):
        return self.forward(x, mask)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class DecoderLayer:
    """Masked self-attention + cross-attention (over the encoder output) +
    FFN, each wrapped in a residual connection and post-norm.
    """
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = LayerNorm(d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm3 = LayerNorm(d_model)

    def __call__(self, x, enc_out, src_mask=None, tgt_mask=None):
        return self.forward(x, enc_out, src_mask, tgt_mask)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x1 = self.norm1(x + self_attn_out)

        cross_out = self.cross_attn(x1, enc_out, enc_out, src_mask)
        x2 = self.norm2(x1 + cross_out)

        ff_out = self.ff(x2)
        x3 = self.norm3(x2 + ff_out)
        return x3

    def backward(self, grad_output):
        """Returns (grad_x, grad_enc_out): the gradient w.r.t. the decoder's
        own input and w.r.t. the encoder output it cross-attended to (the
        latter must be accumulated across all decoder layers by the caller).
        """
        grad_sum3 = self.norm3.backward(grad_output)
        grad_ff_in = self.ff.backward(grad_sum3)
        grad_x2 = grad_sum3 + grad_ff_in

        grad_sum2 = self.norm2.backward(grad_x2)
        grad_q1, grad_k_enc, grad_v_enc = self.cross_attn.backward(grad_sum2)
        grad_x1 = grad_sum2 + grad_q1
        grad_enc_out = grad_k_enc + grad_v_enc

        grad_sum1 = self.norm1.backward(grad_x1)
        grad_q, grad_k, grad_v = self.self_attn.backward(grad_sum1)
        grad_x = grad_sum1 + grad_q + grad_k + grad_v

        return grad_x, grad_enc_out

    def parameters(self):
        return (self.self_attn.parameters() + self.norm1.parameters()
                + self.cross_attn.parameters() + self.norm2.parameters()
                + self.ff.parameters() + self.norm3.parameters())


class Decoder:
    """A stack of `num_layers` DecoderLayers."""
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        self.layers = [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def __call__(self, x, enc_out, src_mask=None, tgt_mask=None):
        return self.forward(x, enc_out, src_mask, tgt_mask)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

    def backward(self, grad_output):
        """Returns (grad_x, grad_enc_out) — grad_enc_out sums the
        cross-attention gradient contributed by every layer, since they all
        read from the same encoder output.
        """
        grad_enc_out_total = 0.0
        for layer in reversed(self.layers):
            grad_output, grad_enc_out = layer.backward(grad_output)
            grad_enc_out_total = grad_enc_out_total + grad_enc_out
        return grad_output, grad_enc_out_total

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class FeatureTokenizer:
    """Turns a (batch, num_features) table of scalars into a (batch,
    num_features, d_model) sequence of "tokens", one per column, so the
    Transformer's attention can be applied to tabular data.

    Each feature gets its own learned affine map scalar -> d_model (a
    per-feature Linear(1, d_model)), since e.g. "Glucose" and "BMI" live on
    different scales and shouldn't share a projection. This follows the
    feature-tokenizer idea from FT-Transformer (Gorishniy et al., 2021).
    """
    def __init__(self, num_features, d_model):
        self.num_features = num_features
        self.d_model = d_model
        self.embeddings = [Linear(1, d_model) for _ in range(num_features)]

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # x: (batch, num_features) -> tokens: (batch, num_features, d_model)
        tokens = [self.embeddings[j](x[:, j:j + 1]) for j in range(self.num_features)]
        return np.stack(tokens, axis=1)

    def backward(self, grad_output):
        # grad_output: (batch, num_features, d_model) -> (batch, num_features)
        grad_cols = [self.embeddings[j].backward(grad_output[:, j, :]) for j in range(self.num_features)]
        return np.concatenate(grad_cols, axis=1)

    def parameters(self):
        params = []
        for embedding in self.embeddings:
            params.append(embedding.W)
            params.append(embedding.b)
        return params


class TabularTransformer:
    """Encoder-decoder Transformer for tabular binary classification.

    There is no natural target *sequence* for a table of features, so the
    decoder side is adapted rather than used for autoregressive generation:
    - Encoder: FeatureTokenizer turns each of the `num_features` columns
      into a d_model token; self-attention lets the encoder learn feature
      interactions (e.g. Glucose x BMI).
    - Decoder: a single learned "query" token (like a [CLS]/object query)
      is fed through masked self-attention (trivial with one token) and
      then cross-attends over the encoder's feature tokens to pool them
      into one summary vector.
    - Head: Linear(d_model, 1) + Sigmoid on the decoder's output token
      produces the class probability, trained with BCE loss.

    This keeps every component (encoder, decoder, cross-attention) faithful
    to the roadmap's "complete encoder-decoder model" while being directly
    usable on the Pima diabetes dataset.
    """
    def __init__(self, num_features, d_model=32, num_heads=4, d_ff=64, num_layers=2):
        self.tokenizer = FeatureTokenizer(num_features, d_model)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers)
        self.query = Parameter(np.random.randn(1, 1, d_model) * 0.02)
        self.head = Linear(d_model, 1)
        self.sigmoid = Sigmoid()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        batch = x.shape[0]
        tokens = self.tokenizer(x)
        enc_out = self.encoder(tokens)

        query = np.repeat(self.query.data, batch, axis=0)  # (batch, 1, d_model)
        dec_out = self.decoder(query, enc_out)

        pooled = dec_out[:, 0, :]  # (batch, d_model)
        logits = self.head(pooled)
        return self.sigmoid(logits)

    def backward(self, grad_output):
        grad_logits = self.sigmoid.backward(grad_output)
        grad_pooled = self.head.backward(grad_logits)
        grad_dec_out = grad_pooled[:, np.newaxis, :]  # (batch, 1, d_model)

        grad_query, grad_enc_out = self.decoder.backward(grad_dec_out)
        grad_query_sum = grad_query.sum(axis=0, keepdims=True)
        self.query.grad = grad_query_sum if self.query.grad is None else self.query.grad + grad_query_sum

        grad_tokens = self.encoder.backward(grad_enc_out)
        grad_x = self.tokenizer.backward(grad_tokens)
        return grad_x

    def parameters(self):
        params = self.tokenizer.parameters() + self.encoder.parameters() + self.decoder.parameters()
        params.append(self.query)
        params.append(self.head.W)
        params.append(self.head.b)
        return params
