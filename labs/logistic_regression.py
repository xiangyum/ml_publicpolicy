
# cross validate + optimization for parameters

from sklearn.linear_model import LogisticRegression
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
    if tp == 0:
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

def cv_LR(X_train = X_train, Y_train = Y_train, pred_threshold = 0.5, penalty_list = ['l1', 'l2'], cvalues_list = [10**-5, 10**-2, 10**-1, 1, 10, 10**2, 10**5], k_folds_num = 5):
   
    # initialize KFolds
    kf = KFold(n_splits = k_folds_num)

    # dictionary of results
    results_LR = {}

    for fold_numb, (train_idx, test_idx) in enumerate(kf.split(X_train)):

        # split into k-folds
        x_split_train, x_split_test = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_split_train, y_split_test = Y_train.iloc[train_idx], Y_train.iloc[test_idx]

        # test for each combination of parameters
        for c_para in cvalues_list:
            for penalty_para in penalty_list:

                #initialize logistic regression object with combination of parameters
                logreg = LogisticRegression(penalty = penalty_para, C = c_para)

                #fit the algorithm with data
                logreg.fit(x_split_train, y_split_train)

                # make predictions. for now, I'll just use the default .predict, which sets the threshold at 0.5
                y_pred = custom_predictions(logreg, threshold = pred_threshold, x_test = x_split_test)

                # write evaluation results to dictionary
                precision_model_key = (c_para,penalty_para,"Precision")
                recall_model_key = (c_para,penalty_para, "Recall")

                # dict.get(key[, default]) this reports the value for the given key, returning a default value of 0 if it's absent
                results_LR[precision_model_key] =  results_LR.get(precision_model_key, 0) + precision_arr(y_pred, y_split_test)/k_folds_num
                results_LR[recall_model_key] = results_LR.get(recall_model_key, 0) + recall_arr(y_pred, y_split_test)/k_folds_num

    #print results
    for model, model_perf in results_LR.items():
        print("Model parameter & evaluation metric: {} | score: {:.5f}".format(model, model_perf))

    return



def custom_predictions(clf, threshold = 0.7, x_test = X_test, y_test = Y_test):
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



def precision(y_pred, y_true):
    tp = 0
    fp = 0
    for i in range(0, len(y_pred)):
        if y_pred.iloc[i,0] == 1:
            if y_true.iloc[i] == 1:
                tp += 1
            else:
                fp += 1
    #calculate & return precision
    return (tp/(tp+fp))

def recall(y_pred, y_true):
    #initialize true positives and false negatives
    tp = 0
    fn = 0

    for i in range(0, len(y_pred)):
        # update tp if i is a true positive
        if y_pred.iloc[i,0] == 1 & y_true.iloc[i] == 1:
            tp += 1

        # update fn if i is a false negative
        if y_pred.iloc[i,0] == 0 & y_true.iloc[i] == 1:
            fn += 1

    return (tp/(tp+fn))


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt 

def plot_roc(pred_y, true_y, randomLine = True):
    # generate false postive rate and true positive rate
    fpr, tpr, thresholds = roc_curve(pred_y, true_y)
    # plot using matplotlib
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr,tpr)
    # plot a red dotted line that represents random guesses
    if randomLine:
        x = [0.0, 1.0]
        plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')
    plt.show()
    return

def print_auc(pred_y, true_y):
    print('AUC Score: {}'.format(roc_auc_score(pred_y, true_y)))
    return

def try_thresholds_range(clf, thresh_list, x_test, y_test):
    # loop through list of threshold probabilities
    for i in thresh_list:     
        # generate predictions
        pred_y = custom_predictions(clf, threshold = i, x_test = x_test, y_test = y_test)
        # turn it into a DF because some of the helper functions I wrote require it
        pred_yDF = pd.DataFrame(pred_y)
        print('Threshold: {}'.format(i))
        print('Recall: {}'.format(recall(pred_yDF, y_test)))
        print('Precision: {}'.format(precision(pred_yDF, y_test)))
        print_auc(pred_y, y_test)
        plot_roc(pred_y, y_test)
    return 




        







#perform k-fold cross validation 
  
    for a in alphas:
        linreg = Ridge(alpha=a)
        linreg.fit(x_split_train, y_split_train)
        y_pred = linreg.predict(x_split_test)
        model_key = (a, ) # this will be a longer tuple for things with more parameters
        results[a] =  results.get(a, 0) + mean_squared_error(y_pred, y_split_test) / splits

for model, model_perf in results.iteritems():
    # the MSE here is meaningless b/c we're fitting random noise to random noise.
    print("Model with params: {} | MSE: {:.2f}".format(model, model_perf))
