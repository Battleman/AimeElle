import numpy as np
from loss import compute_loss
import csv
def baseline(tx_train, y_train, tx_test, y_test, filename):
    w_opt = least_squares(y_train, tx_train)
    loss = compute_loss(y_test, tx_test, w_opt)
    with open(filename, "a") as f:
        writer = csv.writer(f)
        writer.writerow(["baseline", w_opt, loss])


def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)