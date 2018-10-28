########################
####### IMPORTS ########
########################
import numpy as np


########################
######## HELPERS #######
########################
def MSE(y, tx, w):
    return np.sum(np.power(y - np.dot(tx, w), 2)/(2*len(y)))  # MSE


def MAE(y, tx, w):
    return np.sum(np.abs(y - np.dot(tx, w)))/len(y)  # MAE


def RMSE(y, tx, w):
    return np.sqrt(2*MSE(y, tx, w))


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx.dot(w)
    grad = -tx.T.dot(e)/len(y)
    return grad


def calculate_gradient_log(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad


def sigmoid(t):
    """apply sigmoid function on t."""
    return 1/(1+np.exp(-t))


def NLL(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)
########################
###### ASSIGNMENT ######
########################


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = MSE(y, tx, w)
        w = w - gamma*grad
        # print("Step {}, loss is   {}".format(n_iter, loss))
    return (w, loss)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size):
            grad = compute_gradient(y, tx, w)
            w = w - gamma*grad
    loss = MSE(y, tx, w)
    return (w, loss)


def least_squares(y, tx):
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = MSE(y, tx, w)
    return (w, loss)


def ridge_regression(y, tx, lambda_):
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = MSE(y, tx, w)
    return (w, loss)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    def sigmoid(t):
        """apply sigmoid function on t."""
        return 1/(1+np.exp(-t))

    def calculate_gradient(y, tx, w):
        """compute the gradient of loss."""
        pred = sigmoid(tx.dot(w))
        grad = tx.T.dot(pred - y)
        return grad

    def calculate_loss(y, tx, w):
        """compute the cost by negative log likelihood."""
        pred = sigmoid(tx.dot(w))
        loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
        return np.squeeze(- loss)

    def learning_by_gradient_descent(y, tx, w, gamma):
        """
        Do one step of gradient descen using logistic regression.
        Return the loss and the updated w.
        """
        loss = calculate_loss(y, tx, w)
        grad = calculate_gradient(y, tx, w)
        w2 = w - gamma * grad
        print("W went from {} to {}".format(w, w2))
        return loss, w2

    losses = []
    w = initial_w
    for i in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # converge criterion
        print("Step {}, loss is {}".format(i, loss))
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < 1e-8:
            break
    return (w, loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    pass    
