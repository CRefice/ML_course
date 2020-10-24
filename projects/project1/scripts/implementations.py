import numpy as np

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

"""
    Calculate loss via mean squared error.
"""
def calculate_loss(y, tx, w):
    n = y.shape[0]
    e = y - tx @ w
    return (1 / n) * (e.T @ e)

"""
    Calculate the sigmoid function with parameter t,
    which can be an array or a scalar.
"""
def sigmoid(t):
    t = np.clip(t, -500, 500) # to prevent overflows
    exp = np.exp(t)
    return exp / (1 + exp)

"""
    Calculate log loss function via negative log likelihood.
"""
def calculate_log_loss(y, tx, w):
    t = tx @ w
    t = np.clip(t, -500, 500) # to prevent overflows
    a = np.sum(np.log(1 + np.exp(t)))
    b = y.T @ (tx @ w)
    return np.squeeze(a - b)


"""
    Calculate the gradient of the mean squared error loss function.
"""
def calculate_gradient(y, tx, w):
    n = y.shape[0]
    e = y - (tx @ w)
    return (-1 / n) * (tx.T @ e)

"""
    Calculate the gradient of the logistic loss function.
"""
def calculate_log_gradient(y, tx, w):
    sigma = sigmoid(tx @ w)
    return tx.T @ (sigma - y)

"""
    Calculate the hessian of the logistic loss function.
"""
def calculate_hessian(y, tx, w):
    sigma = sigmoid(tx @ w)
    s_rows = np.squeeze(sigma * (1 - sigma))
    s = np.diag(s_rows)
    return tx.T @ s @ tx

"""
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
"""
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


# --------------------------------------------------------------------------------------
# Learning functions
# --------------------------------------------------------------------------------------

"""
    Calculate the weights using full gradient descent with least squares.
"""
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w

    for _ in range(max_iters):
        w = w - gamma * calculate_gradient(y, tx, w)

    loss = calculate_loss(y, tx, w)
    return (w, loss)

"""
    Calculate the weights using stochastic gradient descent with least squares.
"""
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w

    for mini_y, mini_tx in batch_iter(y, tx, 5, max_iters):
        w = w - gamma * calculate_gradient(mini_y, mini_tx, w)

    loss = calculate_loss(y, tx, w)
    return (w, loss)

"""
    Directly calculate optimal weights using the normal equations of least squares.
"""
def least_squares(y, tx):
    tx_t = tx.T
    a = tx_t @ tx
    b = tx_t @ y

    w = np.linalg.solve(a, b)
    loss = calculate_loss(y, tx, w)
    return (w, loss)

"""
    Directly calculate weights using ridge regression.
"""
def ridge_regression(y, tx, lambda_):
    n, d = tx.shape
    lambdap = 2 * n * lambda_
    tx_t = tx.T

    a = tx_t @ tx + lambdap * np.eye(d)
    b = tx_t @ y

    w = np.linalg.solve(a, b)
    loss = calculate_loss(y, tx, w)
    return (w, loss)

"""
    Calculate the weights using full gradient descent with logistic regression.
"""
def logistic_reg_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w

    for _ in range(max_iters):
        w = w - gamma * calculate_log_gradient(y, tx, w)

    loss = calculate_log_loss(y, tx, w)
    return (w, loss)

"""
    Calculate the weights using full gradient descent with penalized logistic regression.
"""
def penalized_logistic_reg_GD(y, tx, initial_w, max_iters, gamma, lambda_):
    w = initial_w

    for _ in range(max_iters):
        gradient = calculate_log_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient

    loss = calculate_log_loss(y, tx, w) + lambda_ * np.squeeze(w.T @ w)
    return (w, loss)

"""
    Calculate the weights using logistic regression with newton's method.
"""
def logistic_reg_newton(y, tx, initial_w, max_iters):
    w = initial_w

    for _ in range(max_iters):
        gradient = calculate_log_gradient(y, tx, w)
        hessian = calculate_hessian(y, tx, w)
        w = w - np.linalg.solve(hessian, gradient)

    loss = calculate_log_loss(y, tx, w)
    return (w, loss)
