import numpy as np


def _calculate_mse(err_vec):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(err_vec**2)


def _calculate_mae(err_vec):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(err_vec))


def compute_loss(y, tx, w, method="mse"):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    y_hat = tx.dot(w)
    err_vec = y - y_hat

    if method == "mse":
        return _calculate_mse(err_vec)
    if method == "mae":
        return _calculate_mae
    raise NotImplementedError
