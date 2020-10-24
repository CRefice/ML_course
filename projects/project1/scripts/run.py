from implementations import *
from proj1_helpers import *


def feature_expansion(tx, degree=10):
    powers = [np.power(tx, i) for i in range(1, degree + 1)]
    return np.concatenate(powers, axis=1)


def normalize(tx, mean, std):
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

useless_features = {
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


def group_by_jet_num(y, tx, jet_num):
    indices = np.where(tx[:, categorical_variable] == jet_num)
    useless = useless_features[jet_num] + ["PRI_jet_num"]
    useless_indices = [feature_names.index(feat) for feat in useless]
    tx_ = np.delete(tx, useless_indices, axis=1)
    return [y[indices], tx_[indices]]


def group_data(y, tx):
    data_grouped = [group_by_jet_num(y, tx, jet_num) for jet_num in range(4)]
    # Create one single group for jet_nums 2 and 3
    data_grouped[2][0] = np.concatenate([data_grouped[2][0], data_grouped[3][0]])
    data_grouped[2][1] = np.concatenate([data_grouped[2][1], data_grouped[3][1]])
    data_grouped.pop(3)
    return data_grouped


lambda_ = 1e-8


def calculate_weights(y, tx):
    mean, std = np.mean(tx, axis=0), np.std(tx, axis=0)
    tx = feature_expansion(normalize(tx, mean, std))
    weights, _ = ridge_regression(y, tx, lambda_)
    return (weights, mean, std)


def predict_labels_grouped(w_by_group, ids, data):
    """Generates class predictions given weights, and a test data matrix"""
    data_grouped = group_data(ids, data)  # List[ (Id, tX)]
    y_pred = np.array([])
    ids_pred = np.array([])
    for group, (w, mean, std) in zip(data_grouped, w_by_group):
        ids, tx = group
        tx = feature_expansion(normalize(tx, mean, std))
        y_pred = np.concatenate([y_pred, tx @ w])
        ids_pred = np.concatenate([ids_pred, ids])
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = -1
    return y_pred, ids_pred


DATA_TRAIN_PATH = "../data/train.csv"
DATA_TEST_PATH = "../data/test.csv"
OUTPUT_PATH = "../data/predictions.csv"

labels, data, _ = load_csv_data(DATA_TRAIN_PATH)
_, tx_test, ids_test = load_csv_data(DATA_TEST_PATH)

w_by_group = [calculate_weights(y, tx) for y, tx in group_data(labels, data)]
y_pred, ids_pred = predict_labels_grouped(w_by_group, ids_test, tx_test)

pred_arr = np.column_stack((ids_pred, y_pred))
sorted_pred = pred_arr[pred_arr[:, 0].argsort()]

create_csv_submission(sorted_pred[:, 0], sorted_pred[:, 1], OUTPUT_PATH)
