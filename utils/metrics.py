# Metrics
import numpy as np

def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of predictions.
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
    Returns:
        float: Accuracy as a percentage.
    """

    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total * 100

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
