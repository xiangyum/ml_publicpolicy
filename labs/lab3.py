import pandas as pd


# If your metric has a metric parameter you need to pass that in via a
# dictionary to metric_params.
knn = KNeighborsClassifier(n_neighbors=10, metric='euclidean', metric_params={'p': 3})

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
knn.predict(x_test)

# try k = 1
knn = KNeighborsClassifier(n_neighbors=10,  metric='manhattan', metric_params={'p': 3})
knn.fit(x_train, y_train)
print(evaluateAccuracy(knn, x_train,y_train))
print(evaluateAccuracy(knn, x_test,y_test))


def evaluateAccuracy(clf,predictDF, truthDF):
    correct_pred = 0
    pred_x = clf.predict(predictDF)
    for i in range(0,len(predictDF)):
        if pred_x[i] == truthDF.iloc[i]:
            correct_pred +=1
    return (correct_pred/len(predictDF))

def try_many_par_KNN():
	#the parameters that I want to test	
	k_list = [3,5,8,10, 13,15,20,25,30,50]
	distance_list = ["euclidean", "manhattan", "chebyshev" ]
	weights_list = ["uniform", "distance"]


	k_rec = []
	dist_rec = []
	weights_rec = []
	tr_acc_rec = []
	test_acc_rec = []

	for index, row in enumerate(k_list):
		for index_d, row_d in enumerate(distance_list):
			for index_w, row_w in enumerate(weights_list):
				knn = KNeighborsClassifier(n_neighbors= row,  metric= row_d, weights = row_w)
				knn.fit(x_train, y_train)
				k_rec.append(row)
				dist_rec.append(row_d)
				weights_rec.append(row_w)
				tr_acc_rec.append(evaluateAccuracy(knn, x_train, y_train))
				test_acc_rec.append(evaluateAccuracy(knn, x_test, y_test))

	finalDF = pd.DataFrame({'k': k_rec,
	                      'dist': dist_rec,
	                      'weights': weights_rec,
	                      'train_acc': tr_acc_rec,
	                      'test_acc': test_acc_rec})

	return finalDF

def try_many_par_DT():

    #the parameters that I want to test
    crit_list = ["gini", "entropy"]
    splitter_list = ["best", "random"]
    maxfeat_list = [None, "auto", "sqrt", "log2", 5, 0.3 ]
    maxdepth_list = [1, 3, 5, 7, 9 ,15 ,20]
    minsplit_list = [2, 3, 4, 5]
    minleaf_list = [1,2,3,4,5]
    maxleaf_list = [None, 2, 3 ,4, 5]

    #create empty lists
    rec_crit = []
    rec_split = []
    rec_maxfeat = []
    rec_maxdepth = []
    rec_minsplit = []
    rec_minleaf = []
    rec_maxleaf = []
    rec_trainP = []
    rec_testP = []

    #loop over every combination of the parameters
    for index_c, crit_par in enumerate(crit_list):
        for index_s, split_par in enumerate(splitter_list):
            for index_f, maxfeat_par in enumerate(maxfeat_list):
                for index_d, maxdepth_par in enumerate(maxdepth_list):
                    for index_ms, minsplit_par in enumerate(minsplit_list):
                        for index_ml, minleaf_par in enumerate(minleaf_list):
                            for index_maxleaf, maxleaf_par in enumerate(maxleaf_list):

                                #initialize classifier
                                dec_tree = DecisionTreeClassifier(criterion = crit_par, splitter = split_par, max_depth = maxdepth_par, min_samples_split = minsplit_par, max_features = maxfeat_par, min_samples_leaf = minleaf_par, max_leaf_nodes = maxleaf_par)
                                dec_tree.fit(x_train, y_train)

                                # evaluate accuracy
                                train_acc = evaluateAccuracy(dec_tree, x_train, y_train)
                                test_acc = evaluateAccuracy(dec_tree, x_test, y_test)

                                #append things to the lists
                                rec_crit.append(crit_par)
                                rec_split.append(split_par)
                                rec_maxfeat.append(maxfeat_par)
                                rec_maxdepth.append(maxdepth_par)
                                rec_minsplit.append(minsplit_par)
                                rec_minleaf.append(minleaf_par)
                                rec_maxleaf.append(maxleaf_par)
                                rec_trainP.append(train_acc)
                                rec_testP.append(test_acc)

    #stich into a pandas DF
    finalDF = pd.DataFrame({"criterion": rec_crit,
                            "splitter": rec_split,
                            "max_features": rec_maxfeat,
                            "max_depth": rec_maxdepth,
                            "min_samples_split": rec_minsplit,
                            "min_samples_leaf": rec_minleaf,
                            "max_leaf_nodes": rec_maxleaf,
                            "prediction_train": rec_trainP,
                            "prediction_test": rec_testP 
                            })

    return finalDF





def try_crit_DT():

    #the parameters that I want to test
    crit_list = ["gini", "entropy"]

    #create empty lists
    rec_crit = []
    rec_trainP = []
    rec_testP = []

    #loop over every combination of the parameters
    for index_c, crit_par in enumerate(crit_list):
        dec_tree = DecisionTreeClassifier(criterion = crit_par)
        dec_tree.fit(x_train, y_train)
        # evaluate accuracy
        train_acc = evaluateAccuracy(dec_tree, x_train, y_train)
        test_acc = evaluateAccuracy(dec_tree, x_test, y_test)

        #append things to the lists
        rec_crit.append(crit_par)
        rec_trainP.append(train_acc)
        rec_testP.append(test_acc)

    #stich into a pandas DF
    finalDF = pd.DataFrame({"criterion": rec_crit,
    "splitter": rec_split,
    "max_features": rec_maxfeat,
    "max_depth": rec_maxdepth,
    "min_samples_split": rec_minsplit,
    "min_samples_leaf": rec_minleaf,
    "max_leaf_nodes": rec_maxleaf,
    "prediction_train": rec_trainP,
    "prediction_test": rec_testP 
    })

    return finalDF





































