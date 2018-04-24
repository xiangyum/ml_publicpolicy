# parameters to optimize for

cvalues_list =  [10**i for i in range(-5, 5)]
kern_list = ["linear", "rbf"]
rbf_gammas_list = [10**i for i in np.arange(0, 1, 0.05)]

from sklearn.svm import SVC
def cv_svm(x_train, y_train, cvalues_list, kern_list, rbf_gammas_list):
    # loop over different C values
    for c_value in cvalues_list:
        #loop over different types of kernels
        for kern in kern_list:
            # for rbf kernels:
            if kern == "rbf":
                for rbf_gam in rbf_gammas_list:
                    clf = SVC(C = c_value, kernel = kern, gamma = rbf_gam, probability = True)





            # for linear kernels
            if kern == "linear":







def find_right_paras_LR_pred_acc(X_train = X_train, Y_train = Y_train, pred_threshold = 0.7, penalty_list = ['l1', 'l2'], cvalues_list = [10**-5, 10**-2, 10**-1, 1, 10, 10**2, 10**5], k_folds_num = 5):
