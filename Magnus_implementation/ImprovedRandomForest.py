# Imports
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from scipy.stats import mode

class ImprovedRandomForest:
    def __init__(self, N_final, m, max_depth, min_samples_leaf, min_samples_split, training_data):

        self.X_train, self.X_reserved, self.y_train, self.y_reserved = split_training_data(data=training_data)

        # split the reserved data into three equal parts
        reserved_size = len(self.X_reserved) // 3
        self.X_res1 = self.X_reserved.iloc[:reserved_size]
        self.X_res2 = self.X_reserved.iloc[reserved_size:2*reserved_size]
        self.X_res3 = self.X_reserved.iloc[2*reserved_size:]
        self.y_res1 = self.y_reserved.iloc[:reserved_size]
        self.y_res2 = self.y_reserved.iloc[reserved_size:2*reserved_size]
        self.y_res3 = self.y_reserved.iloc[2*reserved_size:]
        self.reserved = [(self.X_res1, self.y_res1), (self.X_res2, self.y_res2), (self.X_res3, self.y_res3)]

        self.N_final = N_final
        self.m = m
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split

        # Training a pool of Decision trees with bagging
        forest_pool = make_trees(self.N_final, self.m, self.max_depth, self.min_samples_leaf, self.min_samples_split, self.X_train, self.y_train)
        # Evaluating each tree and sorting the list based on the score
        self.tree_info_sorted = make_tree_sorted(forest_pool, self.reserved)
        # Measures the correlation between the trees with the improved dot product method and determines the deletable trees based on the correlation angle
        self.deletable_mask = correlation_test(self.tree_info_sorted, N_final, self.reserved)
        # Deletes trees primarily on the correlation and secondly on accuracy score if there aren't enough trees marked as deletable to meet the preset
        self.tree_info_sorted = filter_out_trees(self.tree_info_sorted, self.deletable_mask, self.N_final)
        # A list only consisting of the trees excluding excessive information
        self.trees = [tree_information[1] for tree_information in self.tree_info_sorted]

    def get_trees(self):
        return self.trees

    def predict(self, dataset):
        predictions = np.array([tree.predict(dataset) for tree in self.trees])
        majority_vote, _ = mode(predictions, axis=0, keepdims=True)
        return majority_vote.flatten()
    
    def predict_proba(self, dataset):
        all_probas = [t.predict_proba(dataset) for t in self.trees]
        proba_avg = np.mean(np.stack(all_probas, axis=0), axis=0)

        if proba_avg.shape[1] == 1:
            seen = self.trees[0].classes_[0]
            n_samples = proba_avg.shape[0]

            if seen == 1:
                proba_full = np.hstack([np.zeros((n_samples,1)), proba_avg])
            else:
                proba_full = np.hstack([proba_avg, np.zeros((n_samples,1))])
            return proba_full

        return proba_avg



def split_training_data(data, seed=0, train_size=0.8):
    np.random.seed(seed)
    X = data.drop(columns=["Depression"])
    y = data["Depression"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size, random_state=seed, stratify=y)
    return X_train, X_val, y_train, y_val


def make_trees(N_final, m, max_depth, min_samples_leaf, min_samples_split, X_train, y_train):
    """
    Generates the total amount of trees including the
    trees that are going to be deleted later
    """

    # total number of trees to train
    N_pool = int(N_final * (1 + m))

    # list to store the trained trees. 
    forest_pool = []

    for i in range(N_pool):
        X_boot, y_boot = resample(X_train, y_train, random_state=i) # different random_state for each sampling to ensures that each bootstrap sample is different
        clf = DecisionTreeClassifier(random_state=i, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
        clf.fit(X_boot, y_boot)
        forest_pool.append(clf)
    return forest_pool


def average_classification_accuracy(reserved, clf: DecisionTreeClassifier):
    """
    Calculates the average classification accuracy of a decision tree 
    just how the paper describes it.
    """

    accuracy = 0
    for i in reserved:
        accuracy += accuracy_score(i[1], clf.predict(i[0]))

    return (accuracy/3)


def make_tree_sorted(forest_pool, reserved):
    """
    Uses the average_classification_accuracy method to calculate the improved accuracy for
    every tree and later sorts the trees in descending order based on the score.

    Returns the sorted list
    """
    # A list of tuples (tree_index, clf, average_classification_accuracy)
    tree_info = []

    # aplies the average classification accuracy information to each tree in the new list tree_info
    for index, clf in enumerate(forest_pool):
        aca = average_classification_accuracy(reserved, clf)
        tree_info.append((index, clf, aca))

    # Sorts the trees in descending order based on the average classification accuracy
    tree_info_sorted = sorted(tree_info,key=lambda x: x[2], reverse=True)
    
    return tree_info_sorted


def improved_similarity(clf1, clf2):
    """
    Calculates the improved dot product based method to measure the 
    correlation between trees. 

    Takes in two Decision trees and returns an angle representing the correletion

    The paper only considers the feature space meaning the specific features each tree is using, not considering the feature importance.
    """

    # The sets of which features the trees are using
    Wi = set(clf1.tree_.feature)
    Wi.discard(-2) # removing leaves
    Wj = set(clf2.tree_.feature)
    Wj.discard(-2) # removing leaves

    # calculates the numerator of the improved dot product based method in the paper.
    numerator = len(Wi.intersection(Wj))

    # calculates the denominator of the improved dot product based method in the paper.
    denominator = np.sqrt(len(Wi) * len(Wj))

    if denominator == 0:
        return 90  # orthogonal if there's no shared features

    # Cosine similarity normalization
    improved_dot_product = numerator / denominator

    # Clip for numerical stability
    improved_dot_product = np.clip(improved_dot_product, -1.0, 1.0)

    # Puts the whole method for measuring the correlation between two trees. 
    angle_correlation = np.degrees(np.arccos(improved_dot_product))
    
    return angle_correlation


def correlation_test(tree_info_sorted, N_final, reserved):
    """
    Finds the correlation between trees using the improved_similarity() function while also tuning the correlation threshold.
    Returns a mask that can be used to filter out what trees are marked as deletable.
    """

    # The matrix keeping track of the correlations between trees.
    correlation_matrix = np.zeros((len(tree_info_sorted), len(tree_info_sorted)))

    # Finds the correlation between all the trees, adding them to the matrix and marking the 
    # tree with the lowest accuracy among the pairs with too high correlations as deletable with the binary mask marked_mask.
    for i in range(len(tree_info_sorted)):
        for j in range(len(tree_info_sorted)):
            angle = improved_similarity(tree_info_sorted[i][1], tree_info_sorted[j][1])
            correlation_matrix[i, j] = angle
            correlation_matrix[j, i] = angle

    (width, height) = np.shape(correlation_matrix)
    # Hypertune the threshold, tuned differently for each model.
    mean_threshold, min_threshold = find_mean_without_diagonal(width, height, correlation_matrix)
    min_to_mean_threshold_range = mean_threshold - min_threshold
    max_threshold = np.max(correlation_matrix)
    mean_to_max_threshold_range = max_threshold - mean_threshold
    thresholds = [
       mean_threshold + mean_to_max_threshold_range*0.5,
       min_threshold + min_to_mean_threshold_range*0.5,
       mean_threshold,
       mean_threshold + mean_to_max_threshold_range*0.25,
       min_threshold + min_to_mean_threshold_range*0.25
    ]

    best_threshold = 0
    best_score = 0
    best_mask = []

    # Using the grid search method to find the best threshold
    for t, threshold in enumerate(thresholds):

        # The mask representing the trees marked as "deletable" in the tree_info_sorted list
        marked_mask = np.zeros(len(tree_info_sorted))
        tree_info_sorted_temp = tree_info_sorted.copy()

        for i in range(len(tree_info_sorted_temp)):
            for j in range(len(tree_info_sorted_temp)):
                angle = correlation_matrix[i, j]

                if angle < threshold: # if the trees are too correlated
                    if i == j: # skip when it is the same tree
                        continue
                    elif tree_info_sorted_temp[i][2] < tree_info_sorted_temp[j][2]:
                        marked_mask[i] = 1 # mark tree[i] as 1 for "deletable"
                    else:
                        marked_mask[j] = 1 # mark tree[j] as 1 for "deletable"

        # Testing the accuracy of the model with this threshold using the three reserved datasets.
        tree_info_sorted_temp = filter_out_trees(tree_info_sorted_temp, marked_mask, N_final)
        average_accuracy = 0
        trees = [tree_information[1] for tree_information in tree_info_sorted_temp]
        for X, y in reserved:
            predictions = np.array([tree.predict(X) for tree in trees])
            majority_vote, _ = mode(predictions, axis=0, keepdims=True)
            majority_vote = majority_vote.flatten()
            score = accuracy_score(majority_vote, y)
            average_accuracy += score
        average_accuracy = average_accuracy/3
        if average_accuracy > best_score:
            best_mask = marked_mask
            best_threshold = threshold
            best_score = average_accuracy

    return best_mask


def find_mean_without_diagonal(width, height, matrix):
    """
    Finds the mean and min while excluding the times when a tree is compared to itself because that as irrelevant and will pull the mean and min lower than whats correct.
    """
    
    mean = 0
    min = 180
    for i in range(width):
        for j in range(height):
            if i == j:
                continue
            else:
                angle = matrix[i][j]
                mean += angle
                if angle < min:
                    min = angle
    mean = mean / (width*height - width)
    return mean, min


def delete_with_accuracy(tree_info_sorted_temp, preset):
    """
    Sequentially removing trees with the lowest average classification accuracy until the preset number is met
    """
    print("There were too few in the deletable list, deleting based on accuracy score.")
    idx = len(tree_info_sorted_temp) - 1
    while len(tree_info_sorted_temp) > preset and idx >= 0:
        tree_info_sorted_temp.pop(idx)
        idx -= 1
    
    return tree_info_sorted_temp


def filter_out_trees(tree_info_sorted_for_deletion, mask, preset):
    """
    Removes trees primarily based on the deletable trees, but if the preset number isn't met when all the deletable trees are deleted,
    trees will be removed based solely on their accuracy score. 
    """

    mask_temp = list(mask)
    
    idx = len(tree_info_sorted_for_deletion) - 1
    while (len(tree_info_sorted_for_deletion) > preset) and idx >= 0:
        if 1 not in mask_temp[:idx + 1]:
            tree_info_sorted_for_deletion = delete_with_accuracy(tree_info_sorted_for_deletion, preset)
            break
        if mask_temp[idx] == 1:
            tree_info_sorted_for_deletion.pop(idx)
        idx -= 1

    return tree_info_sorted_for_deletion