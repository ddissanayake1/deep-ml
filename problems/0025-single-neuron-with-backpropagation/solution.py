import torch
import torch.nn as nn
import torch.nn.functional as F

def forward(x, w, b):
    z = x @ w + b
    z = F.sigmoid(z)
    return z

def loss_calc(y, y_pred):
    loss = nn.MSELoss()
    return loss(y, y_pred)

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """

    weights = initial_weights.clone().requires_grad_(True)
    bias = torch.tensor(initial_bias, requires_grad = True)

    losses = []

    for epoch in range(epochs):
        y_pred = forward(features, weights, bias)
        loss = loss_calc(labels, y_pred)
        ave_loss = loss.mean()
        losses.append(torch.round(ave_loss, decimals=4).item())
        ave_loss.backward()

        with torch.no_grad():
            weights -= learning_rate*weights.grad
            bias -= learning_rate*bias.grad

            weights.grad.zero_()
            bias.grad.zero_()
    
    return torch.round(weights, decimals=4).tolist(), torch.round(bias,decimals=4).tolist(), losses




    