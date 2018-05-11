# great guide here!: https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import KFold


# important random forest parameters:

# max_features: the max number of features each decision tree is allowed to try.
# increasing max_features generally improves performance, but decreases the speed of the algorithm
max_feats = ["auto", "sqrt", "log2", 0.2]

# n_estimators: the number of decision trees you're going to build.
# higher the better, but of course the higher it is the slower the model is going to be.
num_dtrees_list = [5, 10, 20]

# max_depth: maximum depth of decision trees
depth_list = [3,5,8]

# criterion: gini or entropy
criterion_list = ["gini", "entropy"]



# set the parameter grid
param_grid = {'max_features': max_feats, 'n_estimators' : num_dtrees_list, "max_depth": depth_list, "criterion": criterion_list}
# set the scoring metrics
scoring = {'accuracy': 'accuracy', 'precision': 'precision', "roc": "roc_auc"}

def grid_cv(clf, param_grid, scoring, cv, x_train, y_train):
    # initialize the grid, scoring = a scoring metric or a dictionary of metrics, 
    # refit is necessary when u have a list of scoring metrics, it determines how the gridsearch algorithm decides the best estimator.
    grid = GridSearchCV(clf(), param_grid, scoring = scoring, cv= cv)
    grid.fit(x_train, y_train)

    # call the best classifier:
    grid.best_estimator_

    # see all performances:
    return grid

def grid_cv_mtp(clf = RandomForestClassifier, param_grid, scoring, cv = 5, refit_metric = 'roc'):
    # initialize the grid, scoring = a scoring metric or a dictionary of metrics, 
    # refit is necessary when u have a list of scoring metrics, it determines how the gridsearch algorithm decides the best estimator.
    grid = GridSearchCV(clf(), param_grid, scoring = scoring, cv= cv_num, refit = refit_metric)
    grid.fit(x_train, y_train)

    # call the best classifier:
    grid.best_estimator_

    # see all performances:
    return grid

#