def run_cv_lr(threshold_list):
    results_lr = {}
    penalty_list = ['l1', 'l2']
    cvalues_list = [10**-5, 10**-2, 10**-1, 1, 10, 10**2, 10**5]
    max_feats, num_dtrees_list, depth_list, criterion_list = model_params['max_features'], model_params['n_estimators'],  model_params["max_depth"], model_params["criterion"] 
    for threshold_i in threshold_list:
        for c_para in cvalues_list:
            for penalty_para in penalty_list:
                #initialize logistic regression object with combination of parameters
                logreg = LogisticRegression(penalty = penalty_para, C = c_para)
                logreg.fit(xcv1_train, ycv1_train)
                y_pred = custom_predictions(logreg, threshold = threshold_i, x_test = xcv1_test)                

                prec1 = accuracy_score(y_pred, ycv1_test)
                rec1 = recall_score(y_pred, ycv1_test)
                try:
                    roc1 = roc_auc_score(y_pred, ycv1_test)
                except ValueError:
                    roc1 = 9999999                

                logreg = LogisticRegression(penalty = penalty_para, C = c_para)
                logreg.fit(xcv2_train, ycv2_train)
                y_pred = custom_predictions(logreg, threshold = threshold_i, x_test = xcv2_test)                

                prec2 = accuracy_score(y_pred, ycv2_test)
                rec2 = recall_score(y_pred, ycv2_test)
                try:
                    roc2 = roc_auc_score(y_pred, ycv2_test)
                except ValueError:
                    roc2 = 9999999      

                # intialize the main dictionary key
                model_key = ("LR", c_para, penalty_para)
                prec2 = accuracy_score(y_pred, ycv2_test)
                rec2 = recall_score(y_pred, ycv2_test)
                
                results_lr[model_key] = {}
                # write evaluation results to dictionary
                results_lr[model_key]['Precision']= (prec1 + prec2)/2
                results_lr[model_key]['Recall']= (rec1 + rec2)/2
                results_lr[model_key]['AUC Score'] = (roc1 + roc2)/2

    return results_lr