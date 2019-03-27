""" Methods for working with an ensemble of decision trees """

import numpy as np
import sklearn
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import ensemble as skensemble
from sklearn import tree as sktree
import os
import time

from util import cross_validate_weighted
from load import DataWorker

def get_new_predictions(clf, ld, cutoff):
    results = clf.predict_proba(ld.tests)[:,1]
    new_results = np.zeros(np.shape(results))
    new_results[np.where(results > cutoff)] = 1

    zeros, ones = ld.find_number_classified(new_results)
    ratio = zeros / (zeros + ones)
    print("For cutoff %f, Found: %f are zeroes" % (cutoff, ratio)) #percentage of zeros in the test set

    return new_results, ratio

def get_random_forest(n_estimators=1, max_samples=20000, max_features=20):
    """ Return a BaggingClassifier for a Decision Tree

    Some processing times:
    Linear-Scaling with number of samples and max_features
    max_samples=200K, max_features=200: Takes ~5 minutes
    max_samples=20K, max_features=200: Takes ~30 seconds
    max_samples=200K, max_features=20: Takes ~30 seconds

    """

    simple_tree = skensemble.BaggingClassifier(base_estimator=sktree.DecisionTreeClassifier(), n_estimators=n_estimators, max_samples=max_samples, max_features=max_features)
    return simple_tree

def output_tree(ld, n_estimators=1, max_samples=20000, max_features=200):
    simple_tree = get_random_forest(n_estimators=n_estimators, max_samples=max_samples, max_features=max_features)
    simple_tree.fit(ld.training, ld.targets)
    joblib.dump(simple_tree, "RFclf.pkl")

    return simple_tree

def cross_validate_rf(ld, n_estimators_list=[20], max_samples_list=[20000], max_features_list=[20], saveprefix="cv", weight_positive=9, kfold_n_sets=5):
    """ Cross- Validate a random forest classifier """

    all_input_values = []
    all_clf = []

    for n_estimators in n_estimators_list:
        for max_samples in max_samples_list:
            for max_features in max_features_list:
                all_input_values.append([n_estimators, max_samples, max_features])
                all_clf.append(get_random_forest(n_estimators=n_estimators, max_samples=max_samples, max_features=max_features))

    #training, target, test = ld.get_debug_set()
    training, target, test = ld.get_production_set()

    all_cv_scores = cross_validate_weighted(all_clf, training, target, test, weight_positive=weight_positive, kfold_n_sets=kfold_n_sets)

    np.savetxt("%s_scores.dat" % saveprefix, all_cv_scores)
    np.savetxt("%s_values.dat" % saveprefix, np.array(all_input_values), header="n_estimators, max_samples, max_features", fmt="%d")

if __name__ == "__main__":
    cwd = os.getcwd()
    work_dir = "%s/ensemble_decision_tree" % cwd
    os.makedirs(work_dir, exist_ok=True)

    ld = DataWorker()

    os.chdir(work_dir)
    #simple_tree = output_tree(ld)

#    cross_validate_rf(ld, n_estimators_list=np.arange(10,110,10), max_samples_list=[20000], max_features_list=[200], saveprefix="cv_n_estimators")
    #cross_validate_rf(ld, n_estimators_list=[20], max_samples_list=np.arange(5000,50000,5000), max_features_list=[200], saveprefix="cv_max_samples")
    #cross_validate_rf(ld, n_estimators_list=[20], max_samples_list=[10], max_features_list=np.arange(20,200,20), saveprefix="cv_max_features")

    newtree = output_tree(ld, n_estimators=100, max_samples=20000, max_features=20) #initial bad guess
    results = newtree.predict(ld.tests)
    zeros, ones = ld.find_number_classified(results)
    print("Found: %f are zeroes" % (zeros / (zeros + ones))) #percentage of zeros in the test set

    ld.output_results(results)
    os.chdir(cwd)
