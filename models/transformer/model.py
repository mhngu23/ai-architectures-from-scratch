"""Full Transformer encoder-decoder architecture, shared across 3 different
kinds of data via different "tokenize" front-ends:

- `TabularTransformer` (tabular numeric data): `FeatureTokenizer` projects
  each numeric column into 1 token -> `Encoder` (self-attention across
  features) -> `Decoder` (1 query token cross-attends to pool) ->
  `Linear + Sigmoid`.
- `TextClassifierTransformer` (text data): `TokenEmbedding` +
  `PositionalEncoding` (utils/layers/embedding.py) replace
  `FeatureTokenizer`, the rest of the Encoder/Decoder/head is identical to
  TabularTransformer.
- `Seq2SeqTransformer` (machine translation): same `TokenEmbedding` +
  `PositionalEncoding` front-end as above, but the Decoder generates the
  full target sequence (cross-attending the whole Encoder output, not
  pooled to 1 vector) instead of just classifying.

`EncoderLayer`/`Encoder`/`DecoderLayer`/`Decoder` are the shared building
blocks (self-attention + cross-attention + feed-forward, each branch with
a residual + post-norm LayerNorm) for all 3 models above, built on top of
`MultiHeadAttention` (utils/layers/attention.py) and
`PositionwiseFeedForward` (utils/layers/feedforward.py).
"""
import numpy as np

from utils.layers.linear import Linear, Parameter
from utils.layers.layernorm import LayerNorm
from utils.layers.attention import MultiHeadAttention
from utils.layers.feedforward import PositionwiseFeedForward
from utils.layers.embedding import TokenEmbedding, PositionalEncoding
from utils.layers.dropout import Dropout
from utils.activations.Sigmoid import Sigmoid
from utils.activations.Softmax import Softmax


class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)

    def __call__(self, x, mask=None):
        return self.forward(x, mask)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, x, x, mask)
        attn_out = self.dropout1(attn_out)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        ff_out = self.dropout2(ff_out)
        x = self.norm2(x + ff_out)
        return x

    def backward(self, grad_output):
        grad_sum2 = self.norm2.backward(grad_output)
        grad_ff_out = self.dropout2.backward(grad_sum2)
        grad_ff_in = self.ff.backward(grad_ff_out)
        grad_x1 = grad_sum2 + grad_ff_in

        grad_sum1 = self.norm1.backward(grad_x1)
        grad_attn_out = self.dropout1.backward(grad_sum1)
        grad_q, grad_k, grad_v = self.self_attn.backward(grad_attn_out)
        grad_x = grad_sum1 + grad_q + grad_k + grad_v
        return grad_x

    def parameters(self):
        return (self.self_attn.parameters() + self.norm1.parameters()
                + self.ff.parameters() + self.norm2.parameters())

    def train(self):
        self.dropout1.train()
        self.dropout2.train()

    def eval(self):
        self.dropout1.eval()
        self.dropout2.eval()


class Encoder:
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        self.layers = [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]

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

    def train(self):
        for layer in self.layers:
            layer.train()

    def eval(self):
        for layer in self.layers:
            layer.eval()


class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = Dropout(dropout)

    def __call__(self, x, enc_out, src_mask=None, tgt_mask=None):
        return self.forward(x, enc_out, src_mask, tgt_mask)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        self_attn_out = self.dropout1(self_attn_out)
        x1 = self.norm1(x + self_attn_out)

        cross_out = self.cross_attn(x1, enc_out, enc_out, src_mask)
        cross_out = self.dropout2(cross_out)
        x2 = self.norm2(x1 + cross_out)

        ff_out = self.ff(x2)
        ff_out = self.dropout3(ff_out)
        x3 = self.norm3(x2 + ff_out)
        return x3

    def backward(self, grad_output):
        grad_sum3 = self.norm3.backward(grad_output)
        grad_ff_out = self.dropout3.backward(grad_sum3)
        grad_ff_in = self.ff.backward(grad_ff_out)
        grad_x2 = grad_sum3 + grad_ff_in

        grad_sum2 = self.norm2.backward(grad_x2)
        grad_cross_out = self.dropout2.backward(grad_sum2)
        grad_q1, grad_k_enc, grad_v_enc = self.cross_attn.backward(grad_cross_out)
        grad_x1 = grad_sum2 + grad_q1
        grad_enc_out = grad_k_enc + grad_v_enc

        grad_sum1 = self.norm1.backward(grad_x1)
        grad_self_attn_out = self.dropout1.backward(grad_sum1)
        grad_q, grad_k, grad_v = self.self_attn.backward(grad_self_attn_out)
        grad_x = grad_sum1 + grad_q + grad_k + grad_v

        return grad_x, grad_enc_out

    def parameters(self):
        return (self.self_attn.parameters() + self.norm1.parameters()
                + self.cross_attn.parameters() + self.norm2.parameters()
                + self.ff.parameters() + self.norm3.parameters())

    def train(self):
        self.dropout1.train()
        self.dropout2.train()
        self.dropout3.train()

    def eval(self):
        self.dropout1.eval()
        self.dropout2.eval()
        self.dropout3.eval()


class Decoder:
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        self.layers = [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]

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

    def train(self):
        for layer in self.layers:
            layer.train()

    def eval(self):
        for layer in self.layers:
            layer.eval()


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
    def __init__(self, num_features, d_model=32, num_heads=4, d_ff=64, num_layers=2, dropout=0.1):
        self.tokenizer = FeatureTokenizer(num_features, d_model)
        self.token_dropout = Dropout(dropout)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.query = Parameter(np.random.randn(1, 1, d_model) * 0.02)
        self.head = Linear(d_model, 1)
        self.sigmoid = Sigmoid()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        batch = x.shape[0]
        tokens = self.tokenizer(x)
        tokens = self.token_dropout(tokens)
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
        grad_tokens = self.token_dropout.backward(grad_tokens)
        grad_x = self.tokenizer.backward(grad_tokens)
        return grad_x

    def parameters(self):
        params = self.tokenizer.parameters() + self.encoder.parameters() + self.decoder.parameters()
        params.append(self.query)
        params.append(self.head.W)
        params.append(self.head.b)
        return params

    def train(self):
        self.token_dropout.train()
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.token_dropout.eval()
        self.encoder.eval()
        self.decoder.eval()


def make_causal_mask(seq_len):
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask[np.newaxis, np.newaxis, :, :]


def make_padding_mask(token_ids, pad_id=0):
    """Mask covering `<pad>` positions (sentences in a batch have
    different lengths so they must be padded) — used by encoder
    self-attention and decoder cross-attention so they don't attend to
    pad tokens."""
    mask = (token_ids != pad_id).astype(np.float64)
    return mask[:, np.newaxis, np.newaxis, :]


def make_target_mask(token_ids, pad_id=0):
    """Mask for decoder self-attention: combines the causal mask (can't
    look at future tokens) with the padding mask (can't attend to
    `<pad>` within the target sequence itself)."""
    seq_len = token_ids.shape[1]
    causal = make_causal_mask(seq_len)
    padding = make_padding_mask(token_ids, pad_id)
    return causal * padding


class TextClassifierTransformer:
    def __init__(self, vocab_size, d_model=32, num_heads=4, d_ff=64, num_layers=2, max_len=64, dropout=0.1):
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.embed_dropout = Dropout(dropout)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.query = Parameter(np.random.randn(1, 1, d_model) * 0.02)
        self.head = Linear(d_model, 1)
        self.sigmoid = Sigmoid()

    def __call__(self, token_ids, src_mask=None):
        return self.forward(token_ids, src_mask)

    def forward(self, token_ids, src_mask=None):
        batch = token_ids.shape[0]

        x = self.embedding(token_ids)
        x = self.pos_encoding(x)
        x = self.embed_dropout(x)

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
        grad_x = self.embed_dropout.backward(grad_x)
        grad_x = self.pos_encoding.backward(grad_x)
        self.embedding.backward(grad_x)

    def parameters(self):
        params = self.embedding.parameters() + self.encoder.parameters() + self.decoder.parameters()
        params.append(self.query)
        params.append(self.head.W)
        params.append(self.head.b)
        return params

    def train(self):
        self.embed_dropout.train()
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.embed_dropout.eval()
        self.encoder.eval()
        self.decoder.eval()


class Seq2SeqTransformer:
    """Transformer encoder-decoder for machine translation. Unlike
    `TextClassifierTransformer`, the Decoder generates the whole target
    sequence (teacher forcing while training, autoregressive one token at
    a time at inference via `generate`) instead of just pooling to 1
    query token — the Decoder cross-attends the Encoder's full token
    sequence so it keeps information about every source word.

    Source/target use 2 separate `TokenEmbedding`/`PositionalEncoding`
    tables since they're 2 different vocabularies (source language and
    target language).
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=32, num_heads=4,
                 d_ff=64, num_layers=2, max_len=64, pad_id=0, dropout=0.1):
        self.pad_id = pad_id

        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.src_pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.src_embed_dropout = Dropout(dropout)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.tgt_embed_dropout = Dropout(dropout)

        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)

        self.output_proj = Linear(d_model, tgt_vocab_size)
        self.softmax = Softmax()

    def __call__(self, src_ids, tgt_ids):
        return self.forward(src_ids, tgt_ids)

    def _project_output(self, x):
        batch, seq_len, _ = x.shape
        out = self.output_proj(x.reshape(batch * seq_len, -1))
        return out.reshape(batch, seq_len, -1)

    def _project_output_backward(self, grad, batch, seq_len):
        grad_in = self.output_proj.backward(grad.reshape(batch * seq_len, -1))
        return grad_in.reshape(batch, seq_len, -1)

    def encode(self, src_ids):
        src_mask = make_padding_mask(src_ids, self.pad_id)
        x = self.src_pos_encoding(self.src_embedding(src_ids))
        x = self.src_embed_dropout(x)
        return self.encoder(x, src_mask), src_mask

    def forward(self, src_ids, tgt_ids):
        self.batch, self.tgt_len = tgt_ids.shape

        enc_out, src_mask = self.encode(src_ids)
        tgt_mask = make_target_mask(tgt_ids, self.pad_id)

        tgt = self.tgt_pos_encoding(self.tgt_embedding(tgt_ids))
        tgt = self.tgt_embed_dropout(tgt)
        dec_out = self.decoder(tgt, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)

        logits = self._project_output(dec_out)
        return self.softmax(logits)

    def backward(self, grad_output):
        grad_logits = self.softmax.backward(grad_output)
        grad_dec_out = self._project_output_backward(grad_logits, self.batch, self.tgt_len)

        grad_tgt, grad_enc_out = self.decoder.backward(grad_dec_out)
        grad_tgt = self.tgt_embed_dropout.backward(grad_tgt)
        grad_tgt = self.tgt_pos_encoding.backward(grad_tgt)
        self.tgt_embedding.backward(grad_tgt)

        grad_src = self.encoder.backward(grad_enc_out)
        grad_src = self.src_embed_dropout.backward(grad_src)
        grad_src = self.src_pos_encoding.backward(grad_src)
        self.src_embedding.backward(grad_src)

    def parameters(self):
        params = (self.src_embedding.parameters() + self.tgt_embedding.parameters()
                  + self.encoder.parameters() + self.decoder.parameters())
        params.append(self.output_proj.W)
        params.append(self.output_proj.b)
        return params

    def train(self):
        self.src_embed_dropout.train()
        self.tgt_embed_dropout.train()
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.src_embed_dropout.eval()
        self.tgt_embed_dropout.eval()
        self.encoder.eval()
        self.decoder.eval()

    def generate(self, src_ids, bos_id, eos_id, max_len=64):
        """Greedy decoding: generates the target sentence one token at a
        time, stopping once every sentence in the batch has produced
        `eos_id` or `max_len` is reached. Inference only (no gradient
        tracking) -- automatically switches to `eval()` (disabling
        Dropout) for the duration of decoding and restores the previous
        mode afterward, so callers can't forget to do it themselves."""
        was_training = self.src_embed_dropout.training
        self.eval()
        try:
            batch = src_ids.shape[0]
            enc_out, src_mask = self.encode(src_ids)

            tgt_ids = np.full((batch, 1), bos_id, dtype=int)
            for _ in range(max_len - 1):
                tgt_mask = make_target_mask(tgt_ids, self.pad_id)
                tgt = self.tgt_pos_encoding(self.tgt_embedding(tgt_ids))
                tgt = self.tgt_embed_dropout(tgt)
                dec_out = self.decoder(tgt, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)

                logits = self._project_output(dec_out)
                probs = self.softmax(logits)
                next_ids = np.argmax(probs[:, -1, :], axis=-1, keepdims=True)

                tgt_ids = np.concatenate([tgt_ids, next_ids], axis=1)
                if np.all(next_ids == eos_id):
                    break
            return tgt_ids
        finally:
            if was_training:
                self.train()
