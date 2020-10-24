import numpy as np
import inspect


def build_k_indices(length, k_fold):
    """build k indices for k-fold."""
    indices = np.random.permutation(length)
    return np.array_split(indices, k_fold)


def k_fold_iterator(indices):
    for k in range(len(indices)):
        test = indices[k]
        train = np.concatenate(np.delete(indices, k, axis=0))
        yield (test, train)


def normalize(tx, mean, std):
    return (tx - mean) / std
        

def prediction_accuracy(y, tx, w):
    pred = np.squeeze(tx @ w)
    correct = np.round(pred) == y
    return np.count_nonzero(correct) / len(y)


def cross_validation_step(y, tx, test_indices, train_indices, train_function):
    test_tx, test_y = tx[test_indices], y[test_indices]
    train_tx, train_y = tx[train_indices], y[train_indices]
    
    mean, std = np.mean(train_tx), np.std(train_tx)
    train_tx = normalize(train_tx, mean, std)
    test_tx = normalize(test_tx, mean, std)
    
    # train model on training data
    w, _ = train_function(train_y, train_tx)
    # calculate the prediction accuracy for test data
    return prediction_accuracy(test_y, test_tx, w)


def cross_validation(y, tx, k_indices, train_function):
    accs = np.array(
        [
            cross_validation_step(y, tx, test, train, train_function)
            for test, train in k_fold_iterator(k_indices)
        ]
    )
    return [np.mean(accs), np.std(accs)]


def nested_cross_validation(
    y, tx, k_indices, hyperparams, train_function, num_sub_splits=4
):
    scores = np.empty(len(k_indices))
    for (k, (test, trainval)) in enumerate(k_fold_iterator(k_indices)):
        inner_folds = build_k_indices(len(trainval), num_sub_splits)
        # Each column is a hyperparameter, each row a subfold
        results = np.array(
            [
                [
                    cross_validation_step(
                        y, tx, val, train, lambda y, tx: train_function(y, tx, param)
                    )
                    for param in hyperparams
                ]
                for val, train in k_fold_iterator(inner_folds)
            ]
        )
        average_by_hyperparam = np.mean(results, axis=1)
        best_idx = np.argmax(average_by_hyperparam)
        best_hyperparam = hyperparams[best_idx]
        print(f"Best hyperparam for iteration {k}: {best_hyperparam}")
        scores[k] = cross_validation_step(
            y, tx, test, trainval, lambda y, tx: train_function(y, tx, best_hyperparam)
        )
    print()
    return [np.mean(scores), np.std(scores)]
