########################
####### IMPORTS ########
########################
import numpy as np
import itertools
from proj1_helpers import *

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


def multinomial_partitions(n, k):
    """returns an array of length k sequences of integer partitions of n"""
    print("Multinomial partition for ({},{})".format(n,k))
    nparts = itertools.combinations(range(1, n+k), k-1)
    tmp = [(0,) + p + (n+k,) for p in nparts]
    sequences = np.diff(tmp) - 1
    return sequences[::-1]  # reverse the order


def build_multinomial_crossterms(tx, degree):
    '''Make multinomial feature matrix'''
    order = np.arange(degree)+1
    Xtmp = np.ones_like(tx[:, 0])
    print("Building multinomial feature matrix")
    for ord in order:
        # print("Order {}".format(ord))
        if ord == 1:
            fstmp = tx
        else:
            pwrs = multinomial_partitions(ord, tx.shape[1])
            fstmp = np.column_stack(
                (np.prod(tx**pwrs[i, :], axis=1) for i in range(pwrs.shape[0])))

        Xtmp = np.column_stack((Xtmp, fstmp))
    return Xtmp


def build_poly(x, degree):
    # polynomial basis function:
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data

    phi = np.vander(x, N=degree+1, increasing=True)
    # print("Phi of {} degr {} is {}".format(x,degree,phi))
    return phi


def build_multinomial(tx, degree, important_features, other_features):
    """Build a polynomial list of a certain degree
    
    Arguments:
        tx {np.ndarray} -- 2D numpy array. First dimension is rows, second dimension is features
        degree {int} -- Maximum degree to apply
        important_features {iterable} -- list or tuple of ints. Indices of the "important" columns
        other_features {iterable} -- list or tuple of ints. Indices of the "other" columns
    
    Returns:
        [type] -- [description]
    """
    poly_other = []
    for feature in other_features:
        data_col = tx[:, feature]
        poly_other.append(build_poly(data_col, degree)[:, 1:])
    poly_other = np.concatenate(poly_other, axis=1)
    # print("Poly other is {}".format(poly_other))
    # build polinomial with cross terms as well for important features
    poly_important = build_multinomial_crossterms(
        tx[:, important_features], degree)

    return np.column_stack((poly_important, poly_other))

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
    weights = initial_w
    for _ in range(max_iters):
        rand_index = np.random.randint(y.shape)
        y_batch, tx_batch = y[rand_index], tx[rand_index]
        grad = compute_gradient(y_batch, tx_batch, weights)
        weights = weights - gamma*grad
    loss = MSE(y, tx, weights)
    return (weights, loss)


def least_squares(y, tx):
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = MSE(y, tx, w)
    return (w, loss)

def apply_ridge_regression(y, x, degrees, lambdas, kfold, seed, important_features, other_features):
    lendegrees = len(degrees)
    lenlambdas = len(lambdas)
    lm = np.zeros((lendegrees, lenlambdas))
    wm = []
    row = len(y)
    size = np.floor(row/kfold)
    np.random.seed(seed)
    permutation = np.random.permutation(row)
    for i in range(kfold):
        indices = [permutation[i * size: (i + 1) * size]]
    k = np.array(indices)
    for i, degree in enumerate(degrees):
        multi = build_multinomial(x, degree, important_features, other_features)
        wl = []
        for j, l in enumerate(lambdas):
            w1 = []
            loss1 = []

            for n in range(kfold):
                test1 = y[k[n]]
                train1 = np.delete(y, k[n])
                test2 = multi(k[l])
                train2 = np.delete(multi, k[l], 0)
                w2 = ridge_regression(train1, train2, l)
                loss2 = len(np.where(predict_labels(w2, test2) != test1)[0])/len(test1)
                w1.append(w2)
                loss1.append(loss2)

            wmean = np.asarray(w1).T.mean(1)
            lossmean = np.mean(loss1)
            wl.append(wmean)
            lm[i, j] = lossmean
        wm.append(wl)
    mini = np.argwhere(lm == np.min(lm))

    return wm[mini[0, 0]][mini[0, 1]], degrees[mini[0, 0]]



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

    def calculate_gradient(gradient_y, gradient_tx, gradient_w):
        """compute the gradient of loss."""
        pred = sigmoid(gradient_tx.dot(gradient_w))
        grad = gradient_tx.T.dot(pred - gradient_y)
        return grad

    def calculate_loss(y, loss_tx, loss_w):
        """compute the cost by negative log likelihood."""
        pred = sigmoid(loss_tx.dot(loss_w))
        loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
        return np.squeeze(- loss)

    # def learning_by_gradient_descent(local_y, local_tx, local_w, local_gamma):
    #     """
    #     Do one step of gradient descent using logistic regression.
    #     Return the loss and the updated w.
    #     """
    #     loss = calculate_loss(local_y, local_tx, local_w)
    #     grad = calculate_gradient(local_y, local_tx, local_w)
    #     w2 = local_w - local_gamma * grad
    #     print("W went from {} to {}".format(local_w, w2))
    #     return loss, w2

    # losses = []
    weights = initial_w
    for i in range(max_iters):
        # get loss and update w.
        loss = calculate_loss(y, tx, weights)
        grad = calculate_gradient(y, tx, weights)
        weights = weights - gamma * grad        # converge criterion
        # print("Step {}, loss is {}".format(i, loss))
        # losses.append(loss)
        # if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < 1e-8:
        #     break
    return (weights, loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    pass
