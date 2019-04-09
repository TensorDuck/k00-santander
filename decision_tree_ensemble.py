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

class WeightedPredictionEnsemble(skensemble.BaggingClassifier):

    def __init__(self, **kwargs):
        self.weight_one = kwargs.pop("weight_one", 1)
        super(WeightedPredictionEnsemble, self).__init__(**kwargs)

    def predict(self, X):
        weight_sum = np.zeros(np.shape(X)[0])
        avg_prediction = np.zeros(np.shape(X)[0])
        for clf in self.estimators_:
            results = clf.predict(X)

            weight_sum[np.where(results == 0)] += 1
            weight_sum[np.where(results == 1)] += self.weight_one

            results[np.where(results == 1)] *= self.weight_one
            avg_prediction += results

        final = avg_prediction / weight_sum
        final[np.where(final >= 0.5)] = 1
        final[np.where(final < 1)] = 0

        return final

def get_new_predictions(clf, ld, cutoff):
    results = clf.predict_proba(ld.tests)[:,1]
    new_results = np.zeros(np.shape(results))
    new_results[np.where(results > cutoff)] = 1

    zeros, ones = ld.find_number_classified(new_results)
    ratio = zeros / (zeros + ones)
    print("For cutoff %f, Found: %f are zeroes" % (cutoff, ratio)) #percentage of zeros in the test set

    return new_results, ratio

def get_random_forest(n_estimators=1, max_samples=20000, max_features=20, class_weight=None, weight_one=1):
    """ Return a BaggingClassifier for a Decision Tree

    Some processing times:
    Linear-Scaling with number of samples and max_features
    max_samples=200K, max_features=200: Takes ~5 minutes
    max_samples=20K, max_features=200: Takes ~30 seconds
    max_samples=200K, max_features=20: Takes ~30 seconds

    """

    #simple_tree = skensemble.BaggingClassifier(base_estimator=sktree.DecisionTreeClassifier(max_features=max_features, class_weight=class_weight), n_estimators=n_estimators, max_samples=max_samples)
    simple_tree = WeightedPredictionEnsemble(base_estimator=sktree.DecisionTreeClassifier(max_features=max_features, class_weight=class_weight), n_estimators=n_estimators, max_samples=max_samples, weight_one=weight_one)
    return simple_tree

def output_tree(ld, n_estimators=1, max_samples=20000, max_features=200, class_weight=None, weight_one=1):
    simple_tree = get_random_forest(n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, class_weight=class_weight, weight_one=weight_one)
    simple_tree.fit(ld.training, ld.targets)
    joblib.dump(simple_tree, "RFclf.pkl")

    return simple_tree

def cross_validate_rf(ld, n_estimators_list=[20], max_samples_list=[20000], max_features_list=[20], saveprefix="cv", weight_positive=9, kfold_n_sets=5, class_weight_list=[1]):
    """ Cross- Validate a random forest classifier """

    all_input_values = []
    all_clf = []

    for n_estimators in n_estimators_list:
        for max_samples in max_samples_list:
            for max_features in max_features_list:
                for weight_one in class_weight_list:
                    all_input_values.append([n_estimators, max_samples, max_features, weight_one])
                    class_weight = {0:1, 1:weight_one}
                    all_clf.append(get_random_forest(n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, class_weight=class_weight, weight_one=weight_one))

    #training, target, test = ld.get_debug_set()
    training, target, test = ld.get_production_set()

    all_cv_scores = cross_validate_weighted(all_clf, training, target, test, weight_positive=weight_positive, kfold_n_sets=kfold_n_sets)

    np.savetxt("%s_scores.dat" % saveprefix, all_cv_scores)
    np.savetxt("%s_values.dat" % saveprefix, np.array(all_input_values), header="n_estimators, max_samples, max_features, weight_one", fmt="%d")

if __name__ == "__main__":
    cwd = os.getcwd()
    work_dir = "%s/ensemble_decision_tree" % cwd
    os.makedirs(work_dir, exist_ok=True)

    ld = DataWorker()

    os.chdir(work_dir)
    #simple_tree = output_tree(ld)

#    cross_validate_rf(ld, n_estimators_list=np.arange(10,110,10), max_samples_list=[20000], max_features_list=[200], saveprefix="cv_n_estimators", class_weight={0:1, 1:9}, weight_one=9)
    #cross_validate_rf(ld, n_estimators_list=[20], max_samples_list=np.arange(5000,55000,5000), max_features_list=[20], saveprefix="cv_max_samples", class_weight={0:1, 1:9}, weight_one=9)
    #cross_validate_rf(ld, n_estimators_list=[20], max_samples_list=[20000], max_features_list=[10, 20, 30, 40, 50, 100, 150, 200], saveprefix="cv_max_features", class_weight={0:1, 1:9}, weight_one=9)
    #cross_validate_rf(ld, n_estimators_list=[20], max_samples_list=[30000], max_features_list=[40], saveprefix="cv_weight_one", class_weight_list=[1, 7, 8, 9, 10, 11,15,20])

    newtree = output_tree(ld, n_estimators=200, max_samples=30000, max_features=40, class_weight={0:1, 1:7}, weight_one=7) #initial bad guess
    results = newtree.predict(ld.tests)
    zeros, ones = ld.find_number_classified(results)
    print("Found: %f are zeroes" % (zeros / (zeros + ones))) #percentage of zeros in the test set

    ld.output_results(results)
    os.chdir(cwd)
