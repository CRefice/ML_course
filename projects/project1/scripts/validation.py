import numpy as np

def build_k_indices(y, k_fold):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    indices = np.random.permutation(num_row)
    return [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]


def prediction_accuracy(y, tx, w):
    pred = np.squeeze(tx @ w)
    pred[pred >= 0] = 1
    pred[pred < 0] = -1
    correct = pred == y
    return np.count_nonzero(correct) / len(y)

        
def cross_validation_step(y, tx, k_indices, k, train_function):
    # get k'th subgroup in test, others in train
    test_indices = k_indices[k]
    train_indices = np.delete(k_indices, k, axis=0).flat
    test_tx, test_y = tx[test_indices], y[test_indices]
    train_tx, train_y = tx[train_indices], y[train_indices]
    # train model on training data
    w, loss_train = train_function(train_y, train_tx)
    # calculate the prediction accuracy for test data
    return prediction_accuracy(test_y, test_tx, w)


def cross_validation(y, tx, k_indices, train_function):
    accs = np.array([
        cross_validation_step(y, tx, k_indices, k, train_function)
        for k in range(len(k_indices))
    ])
    return np.mean(accs)


def optimize_hyperparameter(y, tx, hyperparams, k_indices, train_function):
    results = np.array([
        cross_validation(y, tx, k_indices, lambda y, tx: train_function(y, tx, param))
        for param in hyperparams
    ])
    best = np.argmax(results)
#     print(f"Best parameter: {hyperparams[best]} (accuracy: {results[best]})")
    return best, results