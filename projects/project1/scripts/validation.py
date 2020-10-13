from implementations import *

def build_k_indices(y, k_fold):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    indices = np.random.permutation(num_row)
    return [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]

        
def cross_validation_step(y, tx, k_indices, k, train_function):
    # get k'th subgroup in test, others in train
    test_indices = k_indices[k]
    train_indices = np.delete(k_indices, k, axis=0).flat
    test_tx, test_y = tx[test_indices], y[test_indices]
    train_tx, train_y = tx[train_indices], y[train_indices]
    # train model on training data
    w, loss_train = train_function(train_y, train_tx)
    # calculate the loss for train and test data
    loss_test = compute_loss(test_y, test_tx, w)
    return [loss_train, loss_test]


def cross_validation(y, tx, k_indices, train_function):
    losses = [
        cross_validation_step(y, tx, k_indices, k, train_function)
        for k in range(len(k_indices))
    ]
    return np.mean(np.array(losses), axis=0)