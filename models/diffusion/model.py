"""Diffusion model (DDPM — Denoising Diffusion Probabilistic Model, Ho et
al. 2020): `DiffusionSchedule` (fixed beta/alpha/alpha_bar schedule),
`DenoiserMLP` (predicts the noise added to x_0, built from `Linear`/
`SiLU`/`sinusoidal_time_embedding`), and the top-level `DDPM` class that
wires them into the forward noising process (`q_sample`) and the reverse
denoising process (`sample`) — the same pattern as `TabularTransformer`/
`Seq2SeqTransformer` in models/transformer/model.py wiring their own
building blocks together.

Forward process (fixed, no learning):
    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
so a noisy sample at any timestep t can be drawn directly in closed form:
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,  eps ~ N(0, I)

Reverse process (learned): the network eps_theta(x_t, t) predicts the
noise eps that was mixed into x_0 to get x_t. It is trained with the
simplified DDPM objective L_simple = E[|| eps - eps_theta(x_t, t) ||^2]
(plain `MSELoss`, see utils/loss/MSELoss.py). Sampling runs Algorithm 2 of
Ho et al. 2020: starting from x_T ~ N(0, I), iteratively predict and
remove the noise for t = T-1, ..., 0.
"""
import numpy as np

from utils.layers.linear import Linear
from utils.layers.time_embedding import sinusoidal_time_embedding
from utils.activations.SiLU import SiLU


class DiffusionSchedule:
    """Precomputes the fixed beta/alpha/alpha_bar schedule shared by the
    forward process (`DDPM.q_sample`) and the reverse process
    (`DDPM.sample`)."""

    def __init__(self, timesteps=200, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.concatenate([[1.0], self.alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # posterior variance of q(x_{t-1} | x_t, x_0), used for the reverse sampling step
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def __len__(self):
        return self.timesteps


class DenoiserMLP:
    """epsilon_theta(x_t, t) — predicts the noise added to x_0.

        t --sinusoidal_time_embedding--> t_emb --Linear->SiLU->Linear--> t_hidden
        x_t --Linear->SiLU--> x_hidden
        h = x_hidden + t_hidden
        h --Linear->SiLU->Linear--> eps_pred
    """

    def __init__(self, data_dim=2, time_dim=32, hidden_dim=128):
        self.time_dim = time_dim

        self.lin1_x = Linear(data_dim, hidden_dim)
        self.act1_x = SiLU()

        self.lin1_t = Linear(time_dim, hidden_dim)
        self.act1_t = SiLU()
        self.lin2_t = Linear(hidden_dim, hidden_dim)

        self.lin2 = Linear(hidden_dim, hidden_dim)
        self.act2 = SiLU()
        self.lin3 = Linear(hidden_dim, data_dim)

        self.layers = [self.lin1_x, self.lin1_t, self.lin2_t, self.lin2, self.lin3]

    def __call__(self, x_t, t):
        return self.forward(x_t, t)

    def forward(self, x_t, t):
        te = sinusoidal_time_embedding(t, self.time_dim)

        t_hidden = self.lin2_t(self.act1_t(self.lin1_t(te)))
        x_hidden = self.act1_x(self.lin1_x(x_t))

        h = x_hidden + t_hidden
        return self.lin3(self.act2(self.lin2(h)))

    def backward(self, grad_output):
        d_h1_act = self.lin3.backward(grad_output)
        d_h1 = self.act2.backward(d_h1_act)
        d_h = self.lin2.backward(d_h1)

        # h = x_hidden + t_hidden -> the incoming gradient is copied to both branches
        d_x1 = self.act1_x.backward(d_h)
        self.lin1_x.backward(d_x1)

        d_t1_act = self.lin2_t.backward(d_h)
        d_t1 = self.act1_t.backward(d_t1_act)
        self.lin1_t.backward(d_t1)
        # sinusoidal_time_embedding has no learnable parameters -> stop here

    def parameters(self):
        params = []
        for layer in self.layers:
            params.append(layer.W)
            params.append(layer.b)
        return params


class DDPM:
    """Top-level class: wires `DiffusionSchedule` + `DenoiserMLP` into the
    forward process (`q_sample`) and the reverse process (`sample`). See
    models/diffusion/train.py for the training loop and
    models/diffusion/README.md for the math."""

    def __init__(self, data_dim=2, time_dim=32, hidden_dim=128, timesteps=200,
                 beta_start=1e-4, beta_end=0.02):
        self.data_dim = data_dim
        self.schedule = DiffusionSchedule(timesteps, beta_start, beta_end)
        self.model = DenoiserMLP(data_dim, time_dim, hidden_dim)

    def __call__(self, x_t, t):
        return self.model(x_t, t)

    def q_sample(self, x0, t):
        """Forward process: draw x_t ~ q(x_t | x_0) in closed form.

        Args:
            x0: shape (N, data_dim)
            t: integer timesteps, shape (N,)

        Returns:
            x_t: noised sample, shape (N, data_dim)
            eps: the noise that was mixed in, shape (N, data_dim)
        """
        sqrt_ab = self.schedule.sqrt_alphas_cumprod[t][:, None]
        sqrt_1m_ab = self.schedule.sqrt_one_minus_alphas_cumprod[t][:, None]
        eps = np.random.randn(*x0.shape)
        x_t = sqrt_ab * x0 + sqrt_1m_ab * eps
        return x_t, eps

    def backward(self, grad_output):
        self.model.backward(grad_output)

    def parameters(self):
        return self.model.parameters()

    def sample(self, n_samples, return_trajectory=False):
        """Reverse process: generate samples from pure noise (Algorithm 2,
        Ho et al. 2020).

        Args:
            n_samples: how many samples to generate
            return_trajectory: if True, also return every intermediate x_t

        Returns:
            x_0: generated samples, shape (n_samples, data_dim)
            trajectory: list of (n_samples, data_dim) arrays from x_T down
                to x_0 (only if return_trajectory=True)
        """
        x_t = np.random.randn(n_samples, self.data_dim)
        trajectory = [x_t.copy()] if return_trajectory else None

        for t in reversed(range(self.schedule.timesteps)):
            t_batch = np.full(n_samples, t)
            eps_pred = self.model(x_t, t_batch)

            alpha_t = self.schedule.alphas[t]
            alpha_bar_t = self.schedule.alphas_cumprod[t]
            beta_t = self.schedule.betas[t]
            coef = beta_t / np.sqrt(1 - alpha_bar_t)
            mean = (1 / np.sqrt(alpha_t)) * (x_t - coef * eps_pred)

            if t > 0:
                sigma_t = np.sqrt(self.schedule.posterior_variance[t])
                z = np.random.randn(*x_t.shape)
                x_t = mean + sigma_t * z
            else:
                x_t = mean

            if return_trajectory:
                trajectory.append(x_t.copy())

        return (x_t, trajectory) if return_trajectory else x_t
