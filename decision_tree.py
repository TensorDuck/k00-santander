""" Methods for working with a regular decision tree """

import numpy as np
import sklearn
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import tree as sktree
import os
import time

from util import cross_validate_weighted
from load import DataWorker

def make_simple_tree(ld):
    print("Making Tree")
    t1 = time.time()
    simple_tree = sktree.DecisionTreeClassifier(max_depth=100)
    simple_tree.fit(ld.training, ld.targets)
    t2 = time.time()
    print("Making Tree took %f minutes" % ((t2-t1)/60.))
    joblib.dump(simple_tree, "simple_tree.pkl")

    return simple_tree

def output_tree(ld, split_value):
    simple_tree = sktree.DecisionTreeClassifier(max_depth=100, min_samples_split=split_value)
    simple_tree.fit(ld.training, ld.targets)
    joblib.dump(simple_tree, "simple_tree_min_sample_split%d.pkl" % split_value)

    return simple_tree


def cross_validate_tree(ld):
    #cross validate, use stratified KFold and shuffle data
    cv_splitter = StratifiedKFold(n_splits=10, shuffle=True)

    #training, target, test = ld.get_debug_set()
    training, target, test = ld.get_production_set()

    sample_splits = [2, 5, 10, 20, 50, 100, 1000, 2000, 5000, 10000]
    #sample_splits = [2, 5, 10]

    t1 = time.time()
    all_cv_scores = []
    for split_value in sample_splits:
        simple_tree = sktree.DecisionTreeClassifier(max_depth=100, min_samples_split=split_value)
        scores = cross_val_score(simple_tree, training, y=target, cv=cv_splitter)

        all_cv_scores.append(scores)
    t2 = time.time()
    print("Cross Validation took %f minutes" % ((t2-t1) / 60.))
    all_cv_scores = np.array(all_cv_scores)
    np.savetxt("cv_min_split.dat", sample_splits, fmt="%d")
    np.savetxt("cv_scores.dat", all_cv_scores)

def cross_validate_tree_weighted(ld, saveprefix="cv", sample_splits=None, impurity_decrease=None, weight_positive=9, kfold_n_sets=10):
    """ Perform Cross-Validation but apply a weight to positive (1) targets

    Since the training set has many more instances of 0's than 1's (90% 0's), we
    can apply a weight to the 1's (positive targets). This is especially true
    in light of the fact that the test set appears to have many positive
    targets (50% are 1).

    """
    all_trees = []
    if sample_splits is not None:
        allx = sample_splits
        for split_value in sample_splits:
            simple_tree = sktree.DecisionTreeClassifier(max_depth=100, min_samples_split=split_value)
            all_trees.append(simple_tree)
    elif impurity_decrease is not None:
        allx = impurity_decrease
        for i_dec in impurity_decrease:
            simple_tree = sktree.DecisionTreeClassifier(max_depth=100, min_impurity_decrease=i_dec)
            all_trees.append(simple_tree)
    else:
        print("Failure: Must specify set of hyper parameters")
        return


    #training, target, test = ld.get_debug_set()
    training, target, test = ld.get_production_set()

    all_cv_scores = cross_validate_weighted(all_trees, training, target, test, weight_positive=weight_positive, kfold_n_sets=kfold_n_sets)

    np.savetxt("%s_values.dat" % saveprefix, allx)
    np.savetxt("%s_scores.dat" % saveprefix, all_cv_scores)

def run_simple_decision_tree(ld):
    """ First idea: Make Decision Tree without Cross-Validation

    Result: 54% accurate classifier, not great.

    """
    # build the decision simple_tree
    simple_tree = make_simple_tree

    #import the already computed Tree
    simple_tree = joblib.load("simple_tree.pkl")

    # compute decision_tree results
    print ("Computing Results")
    t1 = time.time()
    results = simple_tree.predict(ld.tests)
    t2 = time.time()
    print("Making Prediction took %f minutes" % ((t2-t1)/60.))

    print ("Saving Results")
    t1 = time.time()
    ld.output_results(results)
    t2 = time.time()
    print("Saving Results took %f minutes" % ((t2-t1)/60.))

    a_zeros, a_ones = ld.find_number_classified(ld.targets)
    b_zeros, b_ones = ld.find_number_classified(results)

    print(a_zeros / (a_zeros + a_ones))
    print(b_zeros / (b_zeros + b_ones))

def run_cross_validate_simple_tree_and_select_model(ld):
    """ Second idea: Cross-Validate and select based on accuracy score

    Result: <=50% accuracy, model always outputted 0 due to considerations of
    maximizing accuracy over generality of the model.

    """
    cross_validate_tree(ld)

    tree_1000 = output_tree(ld, 1000)
    tree_6000 = output_tree(ld, 6000)

    results1000 = tree_1000.predict(ld.tests)
    results6000 = tree_6000.predict(ld.tests)

    ld.output_results(results1000, "submission_mss1000.csv")
    ld.output_results(results6000, "submission_mss6000.csv")

if __name__ == "__main__":
    cwd = os.getcwd()
    work_dir = "%s/simple_decision_tree" % cwd
    os.makedirs(work_dir, exist_ok=True)

    ld = DataWorker()

    os.chdir(work_dir)

    #cross_validate_tree_weighted(ld, saveprefix="cv_acc9_sample-count", sample_splits=[2, 5, 10, 20, 50, 100, 200, 500], impurity_decrease=None)
    simple_tree = output_tree(ld, 20)
    os.chdir(cwd)
