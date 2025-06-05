# ai-architectures-from-scratch

A personal collection of core deep learning architectures reimplemented from scratch for study and mastery.  
Includes Transformers, CNNs, RNNs, Autoencoders, GANs, and Diffusion models.

---

## 📚 Purpose

This repository is dedicated to building fundamental deep learning architectures from scratch using minimal libraries (primarily NumPy), to deeply understand their inner workings.

---

## 🧭 Roadmap

Each week is based on a 4-hour time budget.

| Week | Topic                          | Goal |
|------|--------------------------------|------|
| 1    | Project Setup + Core Utils     | Set up repo, implement `Linear`, `ReLU`, and `MSE` from scratch |
| 2    | Optimizers & Training Loop     | Add SGD/Adam, build basic training loop |
| 3-4  | Transformer (Part 1)           | Implement Scaled Dot-Product Attention, Multi-Head Attention |
| 5-6  | Transformer (Part 2)           | Complete encoder-decoder model, run toy training |
| 7-8  | Diffusion Model (Forward)      | Build forward noising process, visualize steps |
| 9-10 | Diffusion Model (Reverse)      | Train denoiser, reconstruct images |
| 11   | CNNs                           | Implement and train simple CNN for image classification |
| 12   | RNN / LSTM / GRU               | Build and test sequence models on toy tasks |
| 13+  | Autoencoders, GANs, ViT, etc.  | Expand to unsupervised and generative models |

---

## 📁 Repository Structure

- `README.md` – This file.
- `requirements.txt` – Python dependencies.
- `utils/` – Common utilities like Linear layers, activation functions, loss functions.
  - `layers` 
  - `loss`
  - `activations`
    - `Relu`
    - `Sigmoid`   
- `tests/` – Simple unit tests for core components.
  - `test_modules.py`
- `notebooks/` – Jupyter notebooks for visualization and exploration.
  - `week1_demo.ipynb`
- `models/`
    - `MLP/` - Standard Multilayer perceptrons
        - `models.py`
    - `transformer/` – Transformer model and training code.
        - `model.py`
        - `train.py`
        - `README.md`
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
