import numpy as np

#--------------------------------------------------------------------------------------
# Helper functions

"""Calculates the loss via MSE"""
def compute_loss(y, tx, w):
    e = y - np.dot(tx,w)
    n = y.shape[0]
    return (1/n) * (np.dot(np.transpose(e),e))


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
        w = w - gamme * compute_gradient(mini_y, mini_tx, w)
    
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


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = compute_loss(y, tx, w)
    grad = compute_log_gradient(y, tx, w)
    w -= gamma * grad
    return loss, w


def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    loss = compute_loss(y, tx, w)
    gradient = compute_log_gradient(y, tx, w)
    hessian = compute_hessian(y, tx, w)
    return loss, gradient, hessian


def learning_by_newton_method(y, tx, w):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss, gradient, hessian = logistic_regression(y, tx, w)
    w -= np.linalg.solve(hessian, gradient)
    return loss, w


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient."""
    num_samples = y.shape[0]
    loss = compute_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = compute_log_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient



def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w -= gamma * gradient
    return loss, w