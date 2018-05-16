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














from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


model_params ={
    RandomForestClassifier: {
    'max_features': ["auto", "sqrt", "log2", 0.2], 
    'n_estimators' : [5, 10, 20, 50, 100, 300, 500], 
    "max_depth": [3,5,8], 
    "criterion": ["gini", "entropy"]
    },
    SVC:{
    "C": [10**i for i in range(-5, 5)],
    "kernel":["linear", "rbf"],
    "gamma": [10**i for i in np.arange(0, 1, 0.05)],
    "probability": [True]
    },
    MultinomialNB:{
    "alpha": [1, 5, 10, 25, 100]
    },
    KNeighborsClassifier:{
    "n_neighbors":[3,5,8,10, 13,15,20,25,30,50],
    "metric": ["euclidean", "manhattan", "chebyshev" ],
    "weights":["uniform", "distance"]
    },
    DecisionTreeClassifier:{
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_depth": [None, "auto", "sqrt", "log2", 5, 0.3 ],
    "min_samples_split": [1, 3, 5, 7, 9 ,15 ,20],
    "max_features": [2, 3, 4, 5],
    "min_samples_leaf": [1,2,3,4,5], 
    "max_leaf_nodes": [None, 2, 3 ,4, 5]
    },
    LogisticRegression:{
    "penalty": ['l1', 'l2'],
    "C": [10**-5, 10**-2, 10**-1, 1, 10, 10**2, 10**5]
    },
    GradientBoostingClassifier:{
    'loss': ["deviance", "exponential"], 
    'learning_rate': [0.01, 0.1, 0.2, 0.3], 
    'n_estimators': [3, 6, 10, 20, 100, 200, 500]
    }
}


def classifier_comparison(model_params, x_train, y_train, eva_metric, cv_num):
    comparison_results = {}
    for model, param_grid in model_params.items():
        # initialize gridsearch object
        grid = GridSearchCV(clf(), param_grid, scoring = eva_metric, cv= cv_num)
        grid.fit(x_train, y_train)
        comparison_results[model] ={}
        comparison_results[model]['cv_results'] = grid.cv_results_
        comparison_results[model]['best_estimator'] = grid.best_estimator_
        comparison_results[model]['best_score'] = grid.best_score_
        comparison_results[model]['best_params'] = grid.best_params_
    return comparison_results


#