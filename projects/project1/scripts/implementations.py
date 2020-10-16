import numpy as np

#--------------------------------------------------------------------------------------
# Helper functions

"""Calculates the loss via MSE"""
def compute_loss(y, tx, w):
    e = y - np.dot(tx,w)
    n = y.shape[0]
    return (1/n) * (np.dot(np.transpose(e),e))


def sigmoid(t):
    """apply the sigmoid function on t."""
    exp = np.exp(t)
    return exp / (1 + exp)


"""Calculates the log loss for logistic regression"""
def compute_log_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    a = np.sum(np.log(1 + np.exp(tx @ w)))
    b = y.T @ (tx @ w)
    return np.squeeze(a - b)


"""Computes the gradient"""
def compute_gradient(y, tx, w):
    n = y.shape[0]
    e = y - np.dot(tx,w)
    return (-1/n) * np.dot(np.transpose(tx),e)


"""Computes the gradient of a logistic function"""
def compute_log_gradient(y, tx, w):
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad


"""Computes the hessian of the loss function."""
def compute_hessian(y, tx, w):
    pred = sigmoid(tx.dot(w))
    pred = np.diag(pred.T[0])
    r = np.multiply(pred, (1-pred))
    return tx.T.dot(r).dot(tx)


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

#--------------------------------------------------------------------------------------
            
"""Computes the prototypes using least-squares with Gradient Descent"""
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    
    for _ in range(max_iters):
        w = w - gamma * compute_gradient(y, tx, w)
        
    loss = compute_loss(y, tx, w)
    
    return (w, loss)


"""Computes the prototypes using least-squares with Stochastic Gradient Descent"""
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    
    for mini_y, mini_tx in batch_iter(y, tx, 1, max_iters):
        w = w - gamma * compute_gradient(mini_y, mini_tx, w)
    
    loss = compute_loss(y, tx, w)
    
    return (w, loss)


"""Directly computes optimal prototypes using the normal equations of least-squares"""
def least_squares(y, tx):
    tx_t = tx.T
    a = tx_t.dot(tx)
    b = tx_t.dot(y)
    
    w = np.linalg.solve(a,b)
    loss = compute_loss(y, tx, w)
    return (w, loss)


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N, D = tx.shape
    lambdap = 2 * N * lambda_
    tx_t = tx.T

    a = tx_t.dot(tx) + lambdap * np.eye(D)
    b = tx_t.dot(y)
    
    w = np.linalg.solve(a,b)
    loss = compute_loss(y, tx, w)
    return (w, loss)


"""Computes the prototypes using logistic regression with gradient descent"""
def logistic_reg_GD(y, tx, initial_w, max_iters, gamma):

    w = initial_w
    
    for _ in range(max_iters):
        w = w - gamma * compute_log_gradient(y, tx, w)   

    loss = compute_log_loss(y, tx, w)

    return (w, loss)


"""Computes the prototypes using penalized logistic regression with gradient descent"""
def penalized_logistic_reg_GD(y, tx, initial_w, max_iters, gamma, lambda_):

    w = initial_w
    
    for _ in range(max_iters):
        gradient = compute_log_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient 

    loss = compute_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))

    return (w, loss)


"""Computes the prototypes using logistic regression with newton method"""
def logistic_reg_newton(y, tx, initial_w, max_iters):

    w = initial_w
    
    for _ in range(max_iters):
        gradient = compute_log_gradient(y, tx, w)
        hessian = compute_hessian(y, tx, w)
        w = w - np.linalg.solve(hessian, gradient)
        
    loss = compute_log_loss(y, tx, w)
    
    return (w, loss)