import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    w = np.zeros(X.shape[1], dtype=float)
    b = np.zeros(1, dtype=float)
    for i in range(steps):
        predictions = _sigmoid(X@w + b) # predict, * would be elementwise
        loss = -np.mean(y * np.log(predictions) + (1-y)*np.log(1-predictions)) # Loss
        w_grad = X.T@(predictions-y)/len(y) # calculate the graidents
        b_grad = np.mean(predictions-y)
        # update
        w = w - (lr*w_grad)
        b = b - (lr*b_grad)
    return w, b