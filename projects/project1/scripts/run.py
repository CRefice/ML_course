import sys

from implementations import *
from proj1_helpers import *


def feature_expansion(tx, degree=10):
    """
    Performs polynomial expansion on the given data matrix.
    Returns the matrix tx concatenated with `degree` powers of itself.
    """
    powers = [np.power(tx, i) for i in range(1, degree + 1)]
    return np.concatenate(powers, axis=1)


def normalize(tx, mean, std):
    """
    Normalizes the matrix tx's columns.
    Mean and std should be computed outside this function,
    since we want to be able to apply the same normalization to the test
    set as that of the training set.
    """
    return (tx - mean) / std


feature_names = [
    "DER_mass_MMC",
    "DER_mass_transverse_met_lep",
    "DER_mass_vis",
    "DER_pt_h",
    "DER_deltaeta_jet_jet",
    "DER_mass_jet_jet",
    "DER_prodeta_jet_jet",
    "DER_deltar_tau_lep",
    "DER_pt_tot",
    "DER_sum_pt",
    "DER_pt_ratio_lep_tau",
    "DER_met_phi_centrality",
    "DER_lep_eta_centrality",
    "PRI_tau_pt",
    "PRI_tau_eta",
    "PRI_tau_phi",
    "PRI_lep_pt",
    "PRI_lep_eta",
    "PRI_lep_phi",
    "PRI_met",
    "PRI_met_phi",
    "PRI_met_sumet",
    "PRI_jet_num",
    "PRI_jet_leading_pt",
    "PRI_jet_leading_eta",
    "PRI_jet_leading_phi",
    "PRI_jet_subleading_pt",
    "PRI_jet_subleading_eta",
    "PRI_jet_subleading_phi",
    "PRI_jet_all_pt",
]

categorical_variable = feature_names.index("PRI_jet_num")

"""
Each value of PRI_jet_num causes a number of features to be undefined.
This map specifies which values correspond to which missing variables.
"""
undefined_features = {
    0: [
        "DER_deltaeta_jet_jet",
        "DER_mass_jet_jet",
        "DER_prodeta_jet_jet",
        "DER_lep_eta_centrality",
        "PRI_jet_leading_pt",
        "PRI_jet_leading_eta",
        "PRI_jet_leading_phi",
        "PRI_jet_subleading_pt",
        "PRI_jet_subleading_eta",
        "PRI_jet_subleading_phi",
        "PRI_jet_all_pt",
    ],
    1: [
        "DER_deltaeta_jet_jet",
        "DER_mass_jet_jet",
        "DER_prodeta_jet_jet",
        "DER_lep_eta_centrality",
        "PRI_jet_subleading_pt",
        "PRI_jet_subleading_eta",
        "PRI_jet_subleading_phi",
    ],
    2: [],
    3: [],
}


def filter_by_jet_num(y, tx, jet_num):
    """
    Given a matrix of labels, training data, and a categorical value of `PRI_jet_num`,
    returns only the rows and labels whose value equals the given one.
    The returned matrix will also be missing features which would be undefined
    for that value of jet_num, along with the PRI_jet_num feature itself.
    """
    group_rows = np.where(tx[:, categorical_variable] == jet_num)

    undefined_names = undefined_features[jet_num] + ["PRI_jet_num"]
    undefined_columns = [feature_names.index(name) for name in undefined_names]
    tx = np.delete(tx, undefined_columns, axis=1)

    return (y[group_rows], tx[group_rows])


def group_by_jet_num(y, tx):
    """
    Returns a list of pairs (y, tx), where each pair is a subset of the given parameters
    grouped their values of PRI_jet_num, with undefined features removed.

    Since both groups 2 and 3 have no undefined variables, in order to increase the size
    of the training set for that group, we merge them into one group.
    Therefore, the returned list should contain three subsets,
    according to jet_num: [0, 1, (2 & 3)]
    """
    groups = [filter_by_jet_num(y, tx, jet_num) for jet_num in range(4)]
    # Merge groups for jet_nums 2 and 3
    groups[2] = (
        np.append(groups[2][0], groups[3][0], axis=0),
        np.append(groups[2][1], groups[3][1], axis=0),
    )
    groups.pop(3)
    return groups


def train_model(y, tx):
    """
    Trains a classification model with the given labels and data set
    using ridge regression with a hard-coded lambda found through hyperparameter
    optimization.

    The training set is first Z-score normalized, and then its features are expanded
    to degree 10.

    Returns a triple containing the computed weights of the model, as well as
    the mean and standard deviation of each feature the training set. This is useful
    since we want to be able to apply the same transformation to the test
    data as to the training set.
    """
    mean, std = np.mean(tx, axis=0), np.std(tx, axis=0)
    tx = feature_expansion(normalize(tx, mean, std), 10)

    lambda_ = 1e-8
    weights, _ = ridge_regression(y, tx, lambda_)
    return (weights, mean, std)


def predict_labels_grouped(group_models, ids, data):
    """
    Given an array of models (weights, mean, stddev), and a list
    of event IDs as well as their corresponding test data,
    Returns a list of predicted labels corresponding to the data.

    We do so by grouping the test data by PRI_jet_num as well.
    We then transform each group with the same mean and stddev as the corresponding
    group in the training set, before obtaining predictions from the model's weights.
    """
    # data_groups: List[(Id, tx)]
    data_groups = group_by_jet_num(ids, data)
    # Grouping data makes it no longer sorted according to ID,
    # so we'll need to re-sort it later. For that reason, we keep track
    # of the ids corresponding to the predictions.
    y_pred = np.array([])
    ids_pred = np.array([])
    for (ids, tx), (w, mean, std) in zip(data_groups, group_models):
        tx = feature_expansion(normalize(tx, mean, std))
        y_pred = np.append(y_pred, tx @ w)
        ids_pred = np.append(ids_pred, ids)

    # Sort predictions by ID
    sorted_indices = ids_pred.argsort()
    y_pred = y_pred[sorted_indices]
    # Clamp predictions to expected range (-1, 1)
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = -1
    return y_pred


if len(sys.argv) < 4:
    print(f"Usage:\n\tpython {sys.argv[0]} TRAIN_FILE TEST_FILE OUTPUT_FILE")
    sys.exit(1)

train_path, test_path, output_path = sys.argv[1:]

print("Loading training data...")
labels, train_data, _ = load_csv_data(train_path)
print("Loading test data...")
_, test_data, ids_test = load_csv_data(test_path)

print("Training model...")
group_models = [train_model(y, tx) for y, tx in group_by_jet_num(labels, train_data)]
print("Generating predictions...")
predictions = predict_labels_grouped(group_models, ids_test, test_data)

print("Saving predictions...")
create_csv_submission(ids_test, predictions, output_path)
