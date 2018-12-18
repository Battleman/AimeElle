import csv

import numpy as np

from loss import compute_loss
from utils import batch_iter


def _least_squares(tx, y):
    """
        Does least squares

        Arguments:
            tx {np.ndarray} -- Reference samples
            y {np.ndarray} -- Reference values

        Returns:
            np.ndarray -- Supposed optimal weights (regression coefficients)
    """

    gram_mat = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(gram_mat, b)


def baseline(tx_train, y_train, tx_test, y_test, filename):
    """
        Baseline method for comparison purpose

        Arguments:
            tx_train {np.ndarray} -- Reference samples
            y_train {np.ndarray} -- Reference values
            tx_test {np.ndarray} -- Test samples
            y_test {np.ndarray} -- Test values
            filename {[type]} -- filename to which append results

        Returns:
            float, np.ndarray -- The loss and the regression coefficients
    """

    w_opt = _least_squares(tx_train, y_train)
    loss = compute_loss(tx_test, y_test, w_opt)
    with open(filename, "a") as f:
        writer = csv.writer(f)
        writer.writerow(["baseline", w_opt, loss])
    return loss, w_opt


def _compute_gradient(tx, y, weights):
    """Compute the gradient."""
    # print("Shapes:\n\ttx: {}\n\ty: {}\n\tweights: {}".format(tx.shape, y.shape, weights.shape))
    y_hat = tx.dot(weights)
    err = y - y_hat
    grad = -tx.T.dot(err) / len(err)
    if isinstance(grad, np.ndarray):
        return grad
    return grad.values


def gradient_descent(tx, y, initial_w, max_iters, gamma, num_batch=1):
    """
        Does a gradient descent (stochastic or complete)

        Arguments:
            tx {np.ndarray} -- Reference samples
            y {np.ndarray} -- Reference values
            initial_w {np.ndarray} -- Initial regression coefficients
            max_iters {int} -- Max number of iterations
            gamma {float} -- Learning rate

        Keyword Arguments:
            num_batch {int} -- Number of batches to use for a stochastic\
                gradient descent (if 1, descent is normal) (default: {1})

        Returns:
            float, np.ndarray -- The loss and the regression coefficients
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    weights = initial_w
    loss_bef = 0
    loss = 0
    for i in range(max_iters):
        for tx_batch, y_batch in batch_iter(tx, y,
                                            batch_size=5000,
                                            num_batches=1):
            grad = _compute_gradient(tx_batch, y_batch, weights)
            weights = weights - gamma*grad  # do a step
            loss_bef = loss
            loss = compute_loss(tx, y, weights)  # compute new loss
            ws.append(weights)
            losses.append(loss)
        if max_iters > 100:
            if i % 100 == 0:
                print("Iteration {}, loss is {}".format(i, loss))
        else:
            print("Iteration {}, loss is {}".format(i, loss))
        if  loss_bef > 0 and loss_bef - loss < 10e-3:
            print("Iteration {}, loss was {}, now is {}, diff = {}".format(i, loss_bef, loss, loss_bef-loss))
            break
    return loss, weights


def ridge_regression(tx, y, lambda_):

    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)
