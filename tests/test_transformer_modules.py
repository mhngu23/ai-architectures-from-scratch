"""Numerical gradient checks and Dropout train/eval-mode checks for the
Transformer building blocks in `models/transformer/model.py`. Run directly
with `python tests/test_transformer_modules.py` from the repo root.
"""
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.transformer.model import TabularTransformer, TextClassifierTransformer, Seq2SeqTransformer
from utils.loss.CrossEntropyLoss import CrossEntropyLoss


def test_seq2seq_dropout_gradient_check():
    """Numerical vs analytical gradient check for `Seq2SeqTransformer`
    with Dropout active (train mode, dropout=0.3). The random mask inside
    each `Dropout` is frozen across the +eps/-eps perturbations by
    reseeding `np.random` right before every forward call, so the finite
    difference compares against the SAME mask realization the analytical
    backward pass used."""
    np.random.seed(0)
    model = Seq2SeqTransformer(
        src_vocab_size=15, tgt_vocab_size=18, d_model=16, num_heads=2,
        d_ff=32, num_layers=2, max_len=12, pad_id=0, dropout=0.3,
    )
    model.train()

    batch = 3
    src_ids = np.random.randint(1, 15, size=(batch, 5))
    tgt_in = np.random.randint(1, 18, size=(batch, 6))
    tgt_out = np.random.randint(1, 18, size=(batch, 6))
    loss_fn = CrossEntropyLoss()

    MASK_SEED = 123

    def compute_loss():
        np.random.seed(MASK_SEED)
        probs = model(src_ids, tgt_in)
        return loss_fn(probs, tgt_out, ignore_index=0)

    loss = compute_loss()
    grad = loss_fn.backward()
    model.backward(grad)

    params = model.parameters()
    print(f"  forward loss: {loss:.6f}")
    print(f"  num params: {len(params)}")
    missing = [i for i, p in enumerate(params) if p.grad is None]
    assert not missing, f"params missing grad: {missing}"
    print("  all params received a gradient: OK")

    eps = 1e-4
    checks = [
        ("output_proj.W[0,0] (near output, after dropout+softmax+CE)", params[-2], (0, 0)),
        ("src_embedding.table[1,0] (deep path: dropout -> encoder -> decoder -> output_proj -> softmax -> CE)", params[0], (1, 0)),
    ]
    for label, p, idx in checks:
        orig = p.data[idx]
        p.data[idx] = orig + eps
        l1 = compute_loss()
        p.data[idx] = orig - eps
        l2 = compute_loss()
        p.data[idx] = orig
        numeric_grad = (l1 - l2) / (2 * eps)
        analytic_grad = p.grad[idx]
        rel_err = abs(numeric_grad - analytic_grad) / (abs(numeric_grad) + abs(analytic_grad) + 1e-8)
        print(f"  {label}")
        print(f"    numeric grad:  {numeric_grad:.8f}")
        print(f"    analytic grad: {analytic_grad:.8f}")
        print(f"    relative error: {rel_err:.2e}  ({'OK' if rel_err < 1e-3 else 'FAIL'})")
        assert rel_err < 1e-3, f"gradient mismatch for {label}: {rel_err:.2e}"

    print("✅ test_seq2seq_dropout_gradient_check passed")


def test_dropout_train_eval_modes():
    """Dropout should make forward output stochastic across repeated
    calls in train mode, and deterministic in eval mode."""
    np.random.seed(0)
    model = TabularTransformer(num_features=4, d_model=8, num_heads=2, d_ff=16, num_layers=1, dropout=0.5)
    x = np.random.randn(3, 4)

    model.train()
    out_a = model(x)
    out_b = model(x)
    train_differs = not np.allclose(out_a, out_b)
    print(f"  train mode: 2 forward calls on same input differ: {train_differs} (expected True)")
    assert train_differs

    model.eval()
    out_c = model(x)
    out_d = model(x)
    eval_identical = np.allclose(out_c, out_d)
    print(f"  eval mode: 2 forward calls on same input identical: {eval_identical} (expected True)")
    assert eval_identical

    print("✅ test_dropout_train_eval_modes passed")


def test_generate_auto_eval_mode():
    """`Seq2SeqTransformer.generate()` must disable Dropout for the
    duration of decoding (deterministic greedy output even if the model
    was left in train mode by the caller) and restore the previous mode
    afterward."""
    np.random.seed(0)
    model = Seq2SeqTransformer(
        src_vocab_size=10, tgt_vocab_size=12, d_model=8, num_heads=2,
        d_ff=16, num_layers=1, max_len=10, pad_id=0, dropout=0.5,
    )
    model.train()
    src = np.random.randint(1, 10, size=(2, 4))

    gen1 = model.generate(src, bos_id=1, eos_id=2, max_len=6)
    still_training = model.src_embed_dropout.training
    print(f"  model still in train() mode after generate() returns: {still_training} (expected True)")
    assert still_training

    gen2 = model.generate(src, bos_id=1, eos_id=2, max_len=6)
    deterministic = np.array_equal(gen1, gen2)
    print(f"  generate() output identical across 2 calls despite train mode: {deterministic} (expected True)")
    assert deterministic

    print("✅ test_generate_auto_eval_mode passed")


def test_text_classifier_smoke():
    """Sanity check that TextClassifierTransformer still runs end-to-end
    with the new dropout parameter wired in."""
    np.random.seed(0)
    model = TextClassifierTransformer(vocab_size=20, d_model=8, num_heads=2, d_ff=16, num_layers=1, max_len=10, dropout=0.5)
    tokens = np.random.randint(1, 20, size=(2, 5))
    model.train()
    out = model(tokens)
    print(f"  output shape: {out.shape} (expected (2, 1))")
    assert out.shape == (2, 1)
    print("✅ test_text_classifier_smoke passed")


if __name__ == "__main__":
    tests = [
        test_seq2seq_dropout_gradient_check,
        test_dropout_train_eval_modes,
        test_generate_auto_eval_mode,
        test_text_classifier_smoke,
    ]
    for test in tests:
        print(f"--- {test.__name__} ---")
        test()
        print()
