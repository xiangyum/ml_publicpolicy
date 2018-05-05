from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import KFold

def cv_multinom_bayes(x_train, y_train, k_folds_num, alpha_list):
    kf = KFold(n_splits = k_folds_num)
    # dictionary of results:
    results_nb = {}
    # split data into k-folds, and loop over each fold
    for fold_numb, (train_idx, test_idx) in enumerate(kf.split(x_train)):
        # split into k-folds
        x_split_train, x_split_test = x_train[train_idx], x_train[test_idx]
        y_split_train, y_split_test = y_train[train_idx], y_train[test_idx]
        #loop over different alpha values
        for alpha_value in alpha_list:
            # intialize classifier and fit it to the split training set
            nb_clf = MultinomialNB(alpha = alpha_value)
            nb_clf.fit(x_split_train, y_split_train)
            # make predictions
            y_pred = nb_clf.predict(x_split_test)
            # intialize the main dictionary key
            model_key = (alpha)
            results_nb[model_key] = {}
            # write evaluation results to dictionary
            results_nb[model_key]['Accuracy']= results_nb[model_key].get('Accuracy', 0) + accuracy_score(y_pred, y_split_test)/k_folds_num
            results_nb[model_key]['Precision']= results_nb[model_key].get('Precision', 0) + precision_score(y_pred, y_split_test)/k_folds_num
            results_nb[model_key]['Recall']= results_nb[model_key].get('Recall', 0) + recall_score(y_pred, y_split_test)/k_folds_num
            results_nb[model_key]['F1']= results_nb[model_key].get('F1', 0) + f1_score(y_pred, y_split_test)/k_folds_num
            results_nb[model_key]['AUC Score']= results_nb[model_key].get('AUC Score', 0) + roc_auc_score(y_pred, y_split_test)/k_folds_num
    #print results
    for model, model_perf in results_nb.items():
        print("Model parameters: {}".format(model, model_perf))
        for eva_metric, eva_value in model_perf.items():
            print("{}: {}".format(eva_metric, eva_value))
        print("————————————————————————————————")
    #return the results, in case I want to do anything with it.
    return results_nb