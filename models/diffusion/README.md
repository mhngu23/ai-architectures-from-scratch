# Diffusion Model (DDPM)

A from-scratch NumPy implementation of DDPM (Denoising Diffusion Probabilistic
Model, [Ho et al. 2020](https://arxiv.org/abs/2006.11239)), applied to toy 2D
data (a "two moons" point cloud) — no autograd, every forward/backward pass is
hand-written and verified with numerical gradient checking
(`tests/test_diffusion_modules.py`).

## Files

- `model.py` — `DiffusionSchedule` (beta/alpha/alpha_bar schedule),
  `DenoiserMLP` (the noise-prediction network `eps_theta(x_t, t)`), and
  `DDPM` (the top-level class wiring both together, exposing the forward
  process `q_sample` and the reverse process `sample`).
- `train.py` — the training loop (`fit`), using the toy datasets from
  `data/toy_datasets.py`. Run with `python -m models.diffusion.train` from
  the repo root. `fit` accepts an existing `optimizer` so a model can keep
  training (with Adam's momentum preserved) across multiple calls on
  different datasets — see `notebooks/week7_demo.ipynb`, sections 7-10.
- `../../data/toy_datasets.py` — 4 toy 2D shapes (`make_two_moons`,
  `make_swiss_roll_2d`, `make_gaussian_mixture`, `make_checkerboard`) plus
  a `TOY_DATASETS` name -> generator dict.
- See `notebooks/week7_demo.ipynb` for a visualized, step-by-step
  walkthrough: sections 1-5 cover a single shape (forward noising process,
  training curve, real-vs-generated samples, denoising trajectory);
  sections 6-10 train across all 4 shapes at once — joint/shuffled
  training vs. one-shape-at-a-time sequential training, and a numeric
  check for catastrophic forgetting in the sequential case.

## The math

**Forward process** (fixed, no learning) gradually adds Gaussian noise to a
data point `x_0` over `T` timesteps according to a fixed variance schedule
`beta_1, ..., beta_T`. With `alpha_t = 1 - beta_t` and
`alpha_bar_t = prod(alpha_1..alpha_t)`, a noisy sample at *any* timestep `t`
can be drawn directly in closed form (no need to iterate step by step):

```
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0, I)
```

This is `DDPM.q_sample`.

**Reverse process** (learned) tries to undo the noising one step at a time.
A network `eps_theta(x_t, t)` (`DenoiserMLP`) is trained to predict the noise
`eps` that was mixed into `x_0` to produce `x_t`, using the simplified DDPM
objective (`L_simple` from the paper — plain MSE, `utils/loss/MSELoss.py`):

```
L_simple = E_{x_0, t, eps} [ || eps - eps_theta(x_t, t) ||^2 ]
```

Sampling (`DDPM.sample`) starts from pure noise `x_T ~ N(0, I)` and runs
Algorithm 2 of the paper: at each step, predict `eps_theta(x_t, t)`, use it to
estimate the mean of `p_theta(x_{t-1} | x_t)`, and add a bit of fresh noise
back in (except at the very last step, `t=0`) — exactly undoing the forward
process one step at a time.

## `DenoiserMLP` architecture

The timestep `t` is embedded with `sinusoidal_time_embedding`
(`utils/layers/time_embedding.py` — the same sin/cos construction as
`PositionalEncoding`, but evaluated for arbitrary integer timesteps). The
data `x_t` and the time embedding are each projected to `hidden_dim`,
summed, and passed through 2 more `Linear`+`SiLU` layers before predicting
`eps`:

```
t --sinusoidal_time_embedding--> t_emb --Linear->SiLU->Linear--> t_hidden
x_t --Linear->SiLU--> x_hidden
h = x_hidden + t_hidden
h --Linear->SiLU->Linear--> eps_pred
```

## Usage

```python
from models.diffusion.model import DDPM
from models.diffusion.train import fit
from data.toy_datasets import make_two_moons

X = make_two_moons(2000)
model = DDPM(data_dim=2, time_dim=32, hidden_dim=128, timesteps=200)
history, optimizer = fit(model, X, n_steps=4000, lr=2e-3)

samples = model.sample(1000)                        # (1000, 2)
samples, trajectory = model.sample(1000, return_trajectory=True)  # + every x_t along the way

# keep training the SAME model on a different dataset, preserving Adam's
# momentum (`optimizer=optimizer`) instead of resetting it — see
# notebooks/week7_demo.ipynb (sections 7-10) for a full multi-shape /
# catastrophic-forgetting example
from data.toy_datasets import make_swiss_roll_2d
history2, optimizer = fit(model, make_swiss_roll_2d(2000), n_steps=1500, optimizer=optimizer)
```

## Possible extensions

- Image data (MNIST): would need `Conv2D`/`ConvTranspose2D` from scratch and
  a small U-Net in place of `DenoiserMLP`.
- DDIM sampling to shorten the number of reverse steps.
- Cosine beta schedule ([Nichol & Dhariwal, 2021](https://arxiv.org/abs/2102.09672)).
- Classifier-free guidance for conditional generation.
