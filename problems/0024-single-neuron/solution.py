import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.
    
    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term
    
    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    # Your code here using PyTorch built-ins:
    # - torch.matmul() for linear combination
    # - torch.sigmoid() for activation
    # - torch.nn.functional.mse_loss() for MSE
    
    x = torch.as_tensor(features, dtype =float)
    y = torch.as_tensor(labels, dtype =float)
    w = torch.as_tensor(weights, dtype =float)
    b = torch.as_tensor(bias, dtype =float)
    
    z = x @ w + b
    y_pred = torch.sigmoid(z)

    mse_loss = F.mse_loss(y, y_pred)

    return (torch.round(y_pred, decimals=4).tolist(), torch.round(mse_loss, decimals=4).item())