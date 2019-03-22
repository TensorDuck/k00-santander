import numpy as np
import sklearn
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import tree as sktree
import os
import time

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

def run_simple_decision_tree():
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

if __name__ == "__main__":
    cwd = os.getcwd()
    work_dir = "%s/simple_decision_tree" % cwd
    os.makedirs(work_dir, exist_ok=True)

    ld = DataWorker()

    os.chdir(work_dir)

    cross_validate_tree(ld)

    os.chdir(cwd)
