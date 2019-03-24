""" Methods for working with a SVM """

import numpy as np
import sklearn
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import svm as sksvm
import os
import time

from util import cross_validate_weighted
from load import DataWorker


def output_svc(ld, c=1.0, kernel="rbf"):
    simple_svm = sksvm.SVC(C=c, kernel=kernel, cache_size=1000, gamma="auto")
    #training, targets, tests = ld.get_normalized_production_set()
    training, targets, tests = ld.get_debug_set()
    simple_svm.fit(training, targets)
    joblib.dump(simple_svm, "simple_svm_k%s_C%f.pkl" % (kernel, c))

    return simple_svm

def cross_validate_svc(ld, c_values=None, kernel="rbf", saveprefix="cv", weight_positive=9, kfold_n_sets=10):
    """ Cross validation with a weighted accuracy prediction """
    all_clf = []
    if c_values is not None:
        allx = c_values
        for c in c_values:
            simple_svm = sksvm.SVC(C=c, kernel=kernel, cache_size=1000, gamma="auto")
            all_clf.append(simple_svm)

    else:
        print("Failure: Must specify set of hyper parameters")
        return

    training, target, test = ld.get_debug_set()
    #training, target, test = ld.get_normalized_production_set()

    all_cv_scores = cross_validate_weighted(all_clf, training, target, test, weight_positive=weight_positive, kfold_n_sets=kfold_n_sets)

    np.savetxt("%s_values.dat" % saveprefix, allx)
    np.savetxt("%s_scores.dat" % saveprefix, all_cv_scores)

if __name__ == "__main__":
    cwd = os.getcwd()
    work_dir = "%s/simple_svm" % cwd
    os.makedirs(work_dir, exist_ok=True)

    ld = DataWorker()

    t1 = time.time()
    os.chdir(work_dir)

    simple_svm = output_svc(ld)

    cross_validate_svc(ld, np.arange(0.1,1.1,0.1))

    training, targets, tests = ld.get_normalized_production_set()
    these_predictions = simple_svm.predict(tests)
    zeros, ones = ld.find_number_classified(these_predictions)
    print("Found: %f are zeroes" % (zeros / (zeros + ones))) #percentage of zeros in the test set

    os.chdir(cwd)
    t2 = time.time()
    "SVM Process took %f minutes" % ((t2-t1)/60.)
