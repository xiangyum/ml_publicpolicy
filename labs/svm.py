# parameters to optimize for

from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def custom_predictions(clf, threshold = 0.7, x_test = X_test):
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

def precision_arr(y_pred, y_true):
    tp = 0
    fp = 0
    for i in range(0, len(y_pred)):
        if y_pred[i] == 1:
            if y_true.iloc[i] == 1:
                tp += 1
            else:
                fp += 1
    #calculate & return precision
    if tp+fp == 0:
        return 0
    else:
        return (tp/(tp+fp))

def recall_arr(y_pred, y_true):
    #initialize true positives and false negatives
    tp = 0
    fn = 0

    for i in range(0, len(y_pred)):
        # update tp if i is a true positive
        if y_pred[i] == 1 & y_true.iloc[i] == 1:
            tp += 1

        # update fn if i is a false negative
        if y_pred[i] == 0 & y_true.iloc[i] == 1:
            fn += 1

    if tp+fn == 0:
        return 0
    else:
        return (tp/(tp+fn))

cvalues_list =  [10**i for i in range(-5, 5)]
kern_list = ["linear", "rbf"]
rbf_gammas_list = [10**i for i in np.arange(0, 1, 0.05)]
k_folds_num = 5

def cv_svm(x_train, y_train, cvalues_list, kern_list, rbf_gammas_list, k_folds_num, pred_threshold):
    # initialize KFolds
    kf = KFold(n_splits = k_folds_num)
    # dictionary of results
    results_SVM = {}
    # split data into k-folds, and loop over each fold
    for fold_numb, (train_idx, test_idx) in enumerate(kf.split(x_train)):
        # split into k-folds
        x_split_train, x_split_test = x_train.iloc[train_idx], x_train.iloc[test_idx]
        y_split_train, y_split_test = y_train.iloc[train_idx], y_train.iloc[test_idx]
        # loop over different C values
        for c_value in cvalues_list:
            #loop over different types of kernels
            for kern in kern_list:
                # for rbf kernels:
                if kern == "rbf":
                    for rbf_gam in rbf_gammas_list:
                        # intialize SVM classifier according to parameters of loop
                        clf = SVC(C = c_value, kernel = kern, gamma = rbf_gam, probability = True)
                        # fit data
                        clf.fit(x_split_train, y_split_train)
                        # make predictions using pre-determined threshold
                        y_pred = custom_predictions(clf, threshold = pred_threshold, x_test = x_split_test)
                        # intialize the main dictionary key
                        model_key = (kern, c_value, rbf_gam)
                        results_SVM[model_key] = {}
                        # write evaluation results to dictionary
                        results_SVM[model_key]['Precision'] = results_SVM[model_key].get('Precision', 0) + precision_arr(y_pred, y_split_test)/k_folds_num
                        results_SVM[model_key]['Recall'] = results_SVM[model_key].get('Recall', 0) + recall_arr(y_pred, y_split_test)/k_folds_num
                        results_SVM[model_key]['AUC Score'] = results_SVM[model_key].get('AUC Score', 0) + roc_auc_score(y_pred, y_split_test)/k_folds_num
                # for linear kernels
                if kern == "linear":
                    # intialize SVM classifier according to parameters of loop
                    clf = SVC(C = c_value, kernel = kern, probability = True)
                    # fit data
                    clf.fit(x_split_train, y_split_train)
                    # make predictions using pre-determined threshold
                    y_pred = custom_predictions(clf, threshold = pred_threshold, x_test = x_split_test)
                    # intialize the main dictionary key
                    model_key = (kern, c_value)
                    results_SVM[model_key] = {}
                    # write evaluation results to dictionary
                    results_SVM[model_key]['Precision'] = results_SVM[model_key].get('Precision', 0) + precision_arr(y_pred, y_split_test)/k_folds_num
                    results_SVM[model_key]['Recall'] = results_SVM[model_key].get('Recall', 0) + recall_arr(y_pred, y_split_test)/k_folds_num
                    results_SVM[model_key]['AUC Score'] = results_SVM[model_key].get('AUC Score', 0) + roc_auc_score(y_pred, y_split_test)/k_folds_num
    #print results
    for model, model_perf in results_SVM.items():
        print("Model parameters: {}".format(model, model_perf))
        for eva_metric, eva_value in model_perf.items():
            print("{}: {}".format(eva_metric, eva_value))
    #return the results, in case I want to do anything with it.
    return





def find_right_paras_LR_pred_acc(X_train = X_train, Y_train = Y_train, pred_threshold = 0.7, penalty_list = ['l1', 'l2'], cvalues_list = [10**-5, 10**-2, 10**-1, 1, 10, 10**2, 10**5], k_folds_num = 5):
