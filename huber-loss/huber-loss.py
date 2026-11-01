import numpy as np

def huber_loss(y_true: list, y_pred: list, delta: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    absolute_error = np.abs(y_true-y_pred)
    result = np.where(absolute_error > delta, delta*(absolute_error-(0.5*delta)), 0.5*(absolute_error**2))
    return float(np.mean(result))