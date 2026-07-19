import numpy as np

from utils.layers.linear import Linear
from utils.activations.ReLU import ReLU


class Encoder:
    """Maps input_dim -> ... -> latent_dim. ReLU is used between hidden
    layers, but the final projection into the latent code has no activation:
    a ReLU there would clamp the latent space to non-negative values only
    (one quadrant of R^latent_dim), which throws away representational
    capacity and makes 2D latent plots collapse onto the axes.
    """
    def __init__(self, input_dim, hidden_dims, latent_dim):
        self.layers = []
        self.activations = []

        dims = [input_dim] + hidden_dims + [latent_dim]
        for i in range(len(dims) - 1):
            is_last = (i == len(dims) - 2)
            self.layers.append(Linear(dims[i], dims[i + 1], activation=None if is_last else 'relu'))
            self.activations.append(None if is_last else ReLU())

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = layer(x)
            if activation:
                x = activation(x)
        return x

    def backward(self, grad_output):
        for layer, activation in reversed(list(zip(self.layers, self.activations))):
            if activation:
                grad_output = activation.backward(grad_output)
            grad_output = layer.backward(grad_output)
        return grad_output

    def parameters(self):
        params = []
        for layer in self.layers:
            params.append(layer.W)
            params.append(layer.b)
        return params


class Decoder:
    """Mirrors the Encoder: latent_dim -> ... -> output_dim. The final layer
    has no activation, since reconstruction targets here are standardized
    (StandardScaler) real-valued features, not values bounded in [0, 1].
    """
    def __init__(self, latent_dim, hidden_dims, output_dim):
        self.layers = []
        self.activations = []

        dims = [latent_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            is_last = (i == len(dims) - 2)
            self.layers.append(Linear(dims[i], dims[i + 1], activation=None if is_last else 'relu'))
            self.activations.append(None if is_last else ReLU())

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = layer(x)
            if activation:
                x = activation(x)
        return x

    def backward(self, grad_output):
        for layer, activation in reversed(list(zip(self.layers, self.activations))):
            if activation:
                grad_output = activation.backward(grad_output)
            grad_output = layer.backward(grad_output)
        return grad_output

    def parameters(self):
        params = []
        for layer in self.layers:
            params.append(layer.W)
            params.append(layer.b)
        return params


class Autoencoder:
    """Encoder-decoder pair trained end-to-end to reconstruct its own input
    (self-supervised: the target is x itself, via MSELoss). The bottleneck
    (`latent_dim` < `input_dim`) forces the encoder to keep only the
    feature interactions needed to reconstruct the data, which is useful
    for dimensionality reduction / visualization and for spotting samples
    that reconstruct poorly (anomalies).
    """
    def __init__(self, input_dim, hidden_dims, latent_dim):
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, list(reversed(hidden_dims)), input_dim)

    def __call__(self, x):
        return self.forward(x)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

    def backward(self, grad_output):
        grad_latent = self.decoder.backward(grad_output)
        return self.encoder.backward(grad_latent)

    def parameters(self):
        return self.encoder.parameters() + self.decoder.parameters()
