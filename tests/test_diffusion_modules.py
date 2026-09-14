"""Numerical gradient check for `DenoiserMLP` and sanity checks for
`DDPM`'s forward/reverse process (models/diffusion/model.py). Run
directly with `python tests/test_diffusion_modules.py` from the repo
root.
"""
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.diffusion.model import DDPM, DenoiserMLP, DiffusionSchedule
from utils.loss.MSELoss import MSELoss


def test_denoiser_gradient_check():
    """Numerical vs analytical gradient check for DenoiserMLP through
    MSELoss, covering both the x_t branch and the time-embedding branch."""
    np.random.seed(0)
    model = DenoiserMLP(data_dim=2, time_dim=8, hidden_dim=16)
    loss_fn = MSELoss()

    batch = 4
    x_t = np.random.randn(batch, 2)
    t = np.random.randint(0, 50, size=batch)
    eps_true = np.random.randn(batch, 2)

    def compute_loss():
        eps_pred = model(x_t, t)
        return loss_fn(eps_pred, eps_true)

    loss = compute_loss()
    model.backward(loss_fn.backward())

    params = model.parameters()
    print(f"  forward loss: {loss:.6f}")
    print(f"  num params: {len(params)}")
    missing = [i for i, p in enumerate(params) if p.grad is None]
    assert not missing, f"params missing grad: {missing}"
    print("  all params received a gradient: OK")

    eps = 1e-4
    checks = [
        ("lin1_x.W[0,0] (x_t branch)", model.lin1_x.W, (0, 0)),
        ("lin1_t.W[0,0] (time-embedding branch)", model.lin1_t.W, (0, 0)),
        ("lin2.W[0,0] (post-merge branch)", model.lin2.W, (0, 0)),
        ("lin3.b[0,0] (output bias)", model.lin3.b, (0, 0)),
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

    print("✅ test_denoiser_gradient_check passed")


def test_schedule_monotonic():
    """alphas_cumprod should decay monotonically towards 0 as more noise
    is mixed in, and sqrt_alphas_cumprod**2 + sqrt_one_minus_alphas_cumprod**2
    should always sum to 1 (they parameterize the same unit-variance mix)."""
    schedule = DiffusionSchedule(timesteps=100)
    is_decreasing = np.all(np.diff(schedule.alphas_cumprod) < 0)
    print(f"  alphas_cumprod strictly decreasing: {is_decreasing} (expected True)")
    assert is_decreasing

    total = schedule.sqrt_alphas_cumprod ** 2 + schedule.sqrt_one_minus_alphas_cumprod ** 2
    sums_to_one = np.allclose(total, 1.0)
    print(f"  sqrt_alphas_cumprod**2 + sqrt_one_minus_alphas_cumprod**2 == 1: {sums_to_one} (expected True)")
    assert sums_to_one

    print("✅ test_schedule_monotonic passed")


def test_ddpm_forward_reverse_shapes():
    """q_sample (forward process) and sample (reverse process) should
    preserve data_dim and always produce finite values."""
    np.random.seed(0)
    model = DDPM(data_dim=2, time_dim=8, hidden_dim=16, timesteps=20)
    X = np.random.randn(10, 2)
    t = np.random.randint(0, 20, size=10)

    x_t, eps = model.q_sample(X, t)
    print(f"  q_sample output shapes: x_t={x_t.shape}, eps={eps.shape} (expected (10, 2) each)")
    assert x_t.shape == X.shape and eps.shape == X.shape
    assert np.all(np.isfinite(x_t))

    samples, trajectory = model.sample(5, return_trajectory=True)
    print(f"  sample output shape: {samples.shape} (expected (5, 2))")
    assert samples.shape == (5, 2)
    assert np.all(np.isfinite(samples))
    print(f"  trajectory length: {len(trajectory)} (expected {model.schedule.timesteps + 1})")
    assert len(trajectory) == model.schedule.timesteps + 1

    print("✅ test_ddpm_forward_reverse_shapes passed")


if __name__ == "__main__":
    tests = [
        test_denoiser_gradient_check,
        test_schedule_monotonic,
        test_ddpm_forward_reverse_shapes,
    ]
    for test in tests:
        print(f"--- {test.__name__} ---")
        test()
        print()
