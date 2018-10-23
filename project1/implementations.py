
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    pass

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    pass

def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)

def ridge_regression(y, tx, lambda_):
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a,b)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    pass

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    pass
