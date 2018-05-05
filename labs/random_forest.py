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

# min_samples_split: the minimum number of samples required to split an internal node
min_split_list = [2,3,5]

# min_samples_leaf: the minimum number of samples required to be at a leaf node:
min_leaf_list = [1,2,3]


def custom_predictions(clf, threshold = 0.7, x_test = X_test):
    # threshold = the probability threshold for something to be a 0.
    # generate array with predicted probabilities
    pred_array = clf.predict_proba(x_test)

    # initialize an empty array for the predictions
    pred_generated = np.array([])

    # predict the first entry
    if pred_array[0][0] >= threshold:
        pred_generated = np.hstack([pred_generated, 0])
    else:
        pred_generated = np.hstack([pred_generated, 1])

    # loops over the rest of the array
    for i in range(1,len(x_test)):
        if pred_array[i][0] >= threshold:
            pred_generated = np.vstack([pred_generated, 0])
        else:
            pred_generated = np.vstack([pred_generated, 1])

    # return an np.array
    return  pred_generated



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import KFold


def grid_cv_RF(x_train, y_train, max_feats, num_dtrees_list, depth_list, criterion_list, k_folds_num):
    kf = KFold(n_splits = k_folds_num)
    # dictionary of results:
    results_rf = {}
    # split data into k-folds, and loop over each fold
    for train_idx, test_idx in kf.split(x_train):
        # split into k-folds
        x_split_train, x_split_test = x_train.iloc[train_idx], x_train.iloc[test_idx]
        y_split_train, y_split_test = y_train.iloc[train_idx], y_train.iloc[test_idx]
        # loop over max_features
        for feature_number in max_feats:
            # loop over n_estimators
            for num_trees in num_dtrees_list:
                #loop over max_depth
                for depth_num in depth_list:
                    #loop over information gain criterion
                    for criterion_choice in criterion_list:
                        rf_clf = RandomForestClassifier(max_features = feature_number, n_estimators = num_trees, max_depth = depth_num, criterion = criterion_choice)
                        rf_clf.fit(x_split_train, y_split_train)
                        # generate prediction using threshold of 0.3 for labels 0
                        y_pred = custom_predictions(rf_clf, threshold = 0.3, x_test = x_split_test)
                        # intialize the main dictionary key
                        model_key = (feature_number, num_trees, depth_num, criterion_choice)
                        # write resuls to dictionary
                        results_rf[model_key] = {}
                        # write evaluation results to dictionary
                        results_rf[model_key]['Accuracy']= results_rf[model_key].get('Accuracy', 0) + accuracy_score(y_pred, y_split_test)/k_folds_num
                        results_rf[model_key]['Precision']= results_rf[model_key].get('Precision', 0) + precision_score(y_pred, y_split_test)/k_folds_num
                        results_rf[model_key]['Recall']= results_rf[model_key].get('Recall', 0) + recall_score(y_pred, y_split_test)/k_folds_num
                        results_rf[model_key]['F1']= results_rf[model_key].get('F1', 0) + f1_score(y_pred, y_split_test)/k_folds_num
    #print results
    for model, model_perf in results_rf.items():
        print("Model parameters: {}".format(model, model_perf))
        for eva_metric, eva_value in model_perf.items():
            print("{}: {}".format(eva_metric, eva_value))
        print("————————————————————————————————")
    #return the results, in case I want to do anything with it.
    return results_rf