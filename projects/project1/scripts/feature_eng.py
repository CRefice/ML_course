import numpy as np
from tqdm import tqdm
from validation import cross_validation, build_k_indices

def backward_selection(training_data, y, x_labels, train_function, k_fold):
    '''Greedy backward selection to iteratively remove features and return a dictionary of the features ranking'''
    k_indices = build_k_indices(y, k_fold)
    rank_features = {}
    nb_tot_features = training_data.shape[1]
    result_features = np.zeros(nb_tot_features)
    to_classify = list(range(nb_tot_features))

    # iterate through set of features and remove the worst at every iteration
    for i in tqdm(range(nb_tot_features-1)):
        loss_per_featset = np.zeros(nb_tot_features-i)
#         print("{} of 30".format(i+1))

        # iterate through every feature which has not yet been classified
        for j, feature in enumerate(to_classify):
            temp_list = [x for x in to_classify if x!=feature]
            temp_data = training_data[:, temp_list]
            loss = cross_validation(y, temp_data, k_indices, train_function)
            loss_per_featset[j] = loss
        
        # register best performance on nb_tot_features-i features
        result_features[i] = np.min(loss_per_featset)
        feat_to_remove = to_classify[np.argmin(loss_per_featset)]
        rank_features[x_labels[feat_to_remove]] = nb_tot_features-i
        to_classify.remove(feat_to_remove)

    #  Finally at the end incorporate the best feature
    rank_features[x_labels[to_classify[0]]] = 1

    return (rank_features, result_features)


def pca(tX):
    'Compute PCA on training data, assumes data is already standardized and that all features are continous (i.e. no categorical)'
    
#     tX = (tX - np.mean(tX, axis=0)) / np.std(tX, axis=0)

    cov_mat = np.cov(tX.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    
    variance_explained = []
    for i in eigen_vals:
         variance_explained.append((i/sum(eigen_vals))*100)

#     return  eigen_vecs.T.dot(tX.T).T, variance_explained
    return  variance_explained, tX.dot(eigen_vecs)