""" Some Custom methods for this project """
import numpy as np
import sklearn
from sklearn.model_selection import cross_val_score, StratifiedKFold

def cross_validate_weighted(all_classifiers, training, target, test, weight_positive=9, kfold_n_sets=10, sample_weight=None):
    cv_splitter = StratifiedKFold(n_splits=kfold_n_sets, shuffle=True)
    split_indices_generator = cv_splitter.split(training, target) # returns a generator

    # make lists of len = kfold_n_sets for training, test, and weights
    kfold_training = []
    kfold_test = []
    kfold_weights = []
    for thing in split_indices_generator:
        kfold_training.append(thing[0])
        kfold_test.append(thing[1])

        #calculate weights for the scoring
        wt = np.ones(np.shape(thing[1]))
        wt[np.where(target[thing[1]] == 1)] = weight_positive
        kfold_weights.append(wt)
    assert len(kfold_training) == kfold_n_sets
    assert len(kfold_test) == kfold_n_sets
    assert len(kfold_weights) == kfold_n_sets

    all_cv_scores = []
    for clf in all_classifiers:
        these_scores = []
        for i_kfold in range(kfold_n_sets):
            this_training = training[kfold_training[i_kfold],:]
            this_target = target[kfold_training[i_kfold]]
            if sample_weight is not None:
                this_weight = sample_weight[kfold_training[i_kfold]]
            else:
                this_weight = None
            clf.fit(this_training, this_target, sample_weight=this_weight)

            test_inputs = training[kfold_test[i_kfold],:]
            test_targets = target[kfold_test[i_kfold]]
            test_weights = kfold_weights[i_kfold]

            test_score = clf.score(test_inputs, test_targets, sample_weight=test_weights)
            these_scores.append(test_score)
        all_cv_scores.append(these_scores)

    all_cv_scores = np.array(all_cv_scores)
    return all_cv_scores
