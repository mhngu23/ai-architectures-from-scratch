# ai-architectures-from-scratch

A personal collection of core deep learning architectures reimplemented from scratch for revision.  
Includes Transformers, CNNs, RNNs, Autoencoders, GANs, and Diffusion models.

---

## 📚 Purpose

This repository is dedicated to building fundamental deep learning architectures from scratch using minimal libraries (primarily NumPy), to deeply understand their inner workings.

---

## 🧭 Roadmap

Each week is based on a 4-hour time budget.

| Week | Topic                          | Goal |
|------|--------------------------------|------|
| 1    | Project Setup + Core Utils     | Set up repo, implement `Linear`, `ReLU`, and `MSE` from scratch ✔️|
| 2    | Optimizers & Training Loop     | Add SGD/Adam, build basic training loop ✔️|
| 3-4  | Transformer (Part 1)           | Implement Scaled Dot-Product Attention, Multi-Head Attention ✔️|
| 5-6  | Transformer (Part 2)           | Complete encoder-decoder model, run toy training ✔️|
| 7-8  | Diffusion Model (Forward)      | Build forward noising process, visualize steps ✔️|
| 9-10 | Diffusion Model (Reverse)      | Train denoiser, reconstruct images ✔️|
| 11   | CNNs                           | Implement and train simple CNN for image classification |
| 12   | RNN / LSTM / GRU               | Build and test sequence models on toy tasks |
| 13+  | Autoencoders, GANs, ViT, etc.  | Expand to unsupervised and generative models |

---

## 📁 Repository Structure

- `README.md` – This file.
- `requirements.txt` – Python dependencies.
- `data/` – Toy datasets, decoupled from any one model so they can be reused across notebooks/scripts.
  - `toy_datasets.py` – `make_two_moons`, `make_swiss_roll_2d`, `make_gaussian_mixture`, `make_checkerboard` (all `(N, 2)`, mean ~0 / std ~1) + a `TOY_DATASETS` name -> generator dict, used by the diffusion model (`models/diffusion/train.py`, `notebooks/week7_demo.ipynb`).
- `utils/` – Common utilities like Linear layers, activation functions, loss functions.
  - `layers`
    - `linear`
    - `layernorm` – LayerNorm (used by the Transformer)
    - `attention` – ScaledDotProductAttention, MultiHeadAttention
    - `feedforward` – PositionwiseFeedForward
    - `embedding` – TokenEmbedding, PositionalEncoding
    - `dropout` – Dropout (regularization; wired into `Encoder`/`Decoder` and all 3 Transformer models, `.train()`/`.eval()` to toggle)
    - `time_embedding` – `sinusoidal_time_embedding`, conditions `DenoiserMLP` (diffusion) on the timestep t
  - `loss`
    - `MSELoss` – also the training objective (`L_simple`) for the diffusion model
    - `BCELoss`
    - `CrossEntropyLoss` – multi-class classification (e.g. next-token prediction in `Seq2SeqTransformer`)
  - `activations`
    - `Relu`
    - `Sigmoid`
    - `Softmax` – paired with `CrossEntropyLoss`
    - `SiLU` – used inside `DenoiserMLP` (diffusion)
  - `optimizers`
    - `Adam`
    - `SGD`   
- `tests/` – Simple unit tests for core components.
  - `test_modules.py`
  - `test_transformer_modules.py` – numerical gradient checks for the encoder/decoder building blocks (incl. `Dropout`) + Dropout train/eval-mode and `generate()` auto-eval checks
  - `test_diffusion_modules.py` – numerical gradient check for `DenoiserMLP`, `DiffusionSchedule` sanity checks, and forward/reverse process shape checks
- `notebooks/` – Jupyter notebooks for visualization and exploration.
  - `week1_demo.ipynb`
  - `week2_demo.ipynb`
  - `week3_demo.ipynb` – MLP baseline on the Pima Indians Diabetes dataset
  - `week4_demo.ipynb` – `FeatureTokenizer` gradient check + `TextClassifierTransformer` toy sentiment classification demo
  - `week5_demo.ipynb` – `Seq2SeqTransformer` toy English -> "unaccented Vietnamese" machine translation demo (teacher forcing, `CrossEntropyLoss`, autoregressive `generate`)
  - `week6_demo.ipynb` – `Seq2SeqTransformer` on the real IWSLT'15 English-Vietnamese corpus (mini-batch training loop, real held-out test sentences)
  - `week7_demo.ipynb` – `DDPM`: (1) on toy 2D "two moons" data - forward noising process visualized step by step, training curve, and reverse-process sampling from pure noise (real-vs-generated comparison + denoising trajectory); (2) across all 4 `data/toy_datasets.py` shapes - joint/shuffled training (all shapes pooled into every minibatch) vs. sequential/continual training (one shape at a time, shuffled curriculum order, same model+optimizer carried across phases), visualizing and numerically scoring (nearest-neighbor coverage) how the sequential model **catastrophically forgets** earlier shapes once training moves on, in contrast to the joint model which keeps covering all 4
- `models/`
    - `MLP/` - Standard Multilayer perceptrons
        - `model.py`
    - `transformer/` – Encoder-decoder Transformer, including a `TabularTransformer` adapted for tabular data (feature tokenizer + encoder self-attention + decoder cross-attention readout), a `TextClassifierTransformer` for text classification, and a `Seq2SeqTransformer` for machine translation (decoder cross-attends the full source sequence + autoregressive `generate` for greedy decoding).
        - `model.py`
    - `diffusion/` – Diffusion model (forward and reverse process).
        - `model.py`
        - `train.py`
        - `README.md`
    - `cnn/` – Convolutional neural network implementation.
        - `model.py`
        - `train.py`
        - `README.md`
    - `rnn_lstm_gru/` – Sequence models: RNN, LSTM, and GRU.
        - `model.py`
        - `train.py`
        - `README.md`

---

## 🚀 Getting Started

```bash
# Clone and set up environment
git clone https://github.com/yourusername/ai-architectures-from-scratch.git
cd ai-architectures-from-scratch
conda create -n dl-study-env python=3.10 -y
conda activate dl-study-env
pip install -r requirements.txt
