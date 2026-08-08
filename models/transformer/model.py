"""Kiến trúc Transformer encoder-decoder đầy đủ, dùng chung cho 2 loại dữ
liệu khác nhau qua 2 tầng "tokenize" riêng biệt:

- `TabularTransformer` (dữ liệu bảng số): `FeatureTokenizer` chiếu mỗi cột
  số thành 1 token -> `Encoder` (self-attention giữa các feature) ->
  `Decoder` (1 query token cross-attend để pool) -> `Linear + Sigmoid`.
- `TextClassifierTransformer` (dữ liệu văn bản): `TokenEmbedding` +
  `PositionalEncoding` (utils/layers/embedding.py) thay cho `FeatureTokenizer`,
  phần Encoder/Decoder/head còn lại giống hệt TabularTransformer.

`EncoderLayer`/`Encoder`/`DecoderLayer`/`Decoder` là các khối dùng chung
(self-attention + cross-attention + feed-forward, mỗi nhánh residual +
LayerNorm post-norm) cho cả 2 model trên, xây trên `MultiHeadAttention`
(utils/layers/attention.py) và `PositionwiseFeedForward` (utils/layers/feedforward.py).
"""
import numpy as np

from utils.layers.linear import Linear, Parameter
from utils.layers.layernorm import LayerNorm
from utils.layers.attention import MultiHeadAttention
from utils.layers.feedforward import PositionwiseFeedForward
from utils.layers.embedding import TokenEmbedding, PositionalEncoding
from utils.activations.Sigmoid import Sigmoid


class EncoderLayer:
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
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        self.layers = [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def __call__(self, x, enc_out, src_mask=None, tgt_mask=None):
        return self.forward(x, enc_out, src_mask, tgt_mask)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

    def backward(self, grad_output):
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
    def __init__(self, num_features, d_model):
        self.num_features = num_features
        self.d_model = d_model
        self.embeddings = [Linear(1, d_model) for _ in range(num_features)]

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        tokens = [self.embeddings[j](x[:, j:j + 1]) for j in range(self.num_features)]
        return np.stack(tokens, axis=1)

    def backward(self, grad_output):
        grad_cols = [self.embeddings[j].backward(grad_output[:, j, :]) for j in range(self.num_features)]
        return np.concatenate(grad_cols, axis=1)

    def parameters(self):
        params = []
        for embedding in self.embeddings:
            params.append(embedding.W)
            params.append(embedding.b)
        return params


class TabularTransformer:
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

        query = np.repeat(self.query.data, batch, axis=0)
        dec_out = self.decoder(query, enc_out)

        pooled = dec_out[:, 0, :]
        logits = self.head(pooled)
        return self.sigmoid(logits)

    def backward(self, grad_output):
        grad_logits = self.sigmoid.backward(grad_output)
        grad_pooled = self.head.backward(grad_logits)
        grad_dec_out = grad_pooled[:, np.newaxis, :]

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


def make_causal_mask(seq_len):
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask[np.newaxis, np.newaxis, :, :]


class TextClassifierTransformer:
    def __init__(self, vocab_size, d_model=32, num_heads=4, d_ff=64, num_layers=2, max_len=64):
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers)
        self.query = Parameter(np.random.randn(1, 1, d_model) * 0.02)
        self.head = Linear(d_model, 1)
        self.sigmoid = Sigmoid()

    def __call__(self, token_ids, src_mask=None):
        return self.forward(token_ids, src_mask)

    def forward(self, token_ids, src_mask=None):
        batch = token_ids.shape[0]

        x = self.embedding(token_ids)
        x = self.pos_encoding(x)

        enc_out = self.encoder(x, src_mask)

        query = np.repeat(self.query.data, batch, axis=0)
        dec_out = self.decoder(query, enc_out, src_mask=src_mask)

        pooled = dec_out[:, 0, :]
        logits = self.head(pooled)
        return self.sigmoid(logits)

    def backward(self, grad_output):
        grad_logits = self.sigmoid.backward(grad_output)
        grad_pooled = self.head.backward(grad_logits)
        grad_dec_out = grad_pooled[:, np.newaxis, :]

        grad_query, grad_enc_out = self.decoder.backward(grad_dec_out)
        grad_query_sum = grad_query.sum(axis=0, keepdims=True)
        self.query.grad = grad_query_sum if self.query.grad is None else self.query.grad + grad_query_sum

        grad_x = self.encoder.backward(grad_enc_out)
        grad_x = self.pos_encoding.backward(grad_x)
        self.embedding.backward(grad_x)

    def parameters(self):
        params = self.embedding.parameters() + self.encoder.parameters() + self.decoder.parameters()
        params.append(self.query)
        params.append(self.head.W)
        params.append(self.head.b)
        return params
