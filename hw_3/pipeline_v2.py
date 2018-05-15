import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score


## Part 1: read/load data

def read_data(fn, filetype = "csv"):
    if filetype == "csv":
        return pd.read_csv(fn)
    if filetype == "excel":
        return pd.read_excel(fn)
    if filetype == "sql":
        return pd.read_sql(fn, con=conn)
    else:
        return print("I only have CSVs at the moment!")


## Part 2: explore data

def show_columns(df):
    return df.columns

def descrip_stats(df):
    return df.describe()

def counts_per_variable(df, x):
	return df.groupby(x).size()

def group_and_describe(df, x):
	return df.groupby(x).describe()

def ctab_percent(df, x, y):
	return pd.crosstab(df.loc[:, x], df.loc[:,y], normalize='index')

def ctab_raw(df, x, y):
	return pd.crosstab(df.loc[:, x], df.loc[:,y])

def basic_hist(df, x, title_text):
	sns.distplot(df[x]).set_title(title_text)
	plt.show()
	return

def basic_scatter(df, x, y, title_text):
	g = sns.lmplot(x, y, data= df)
	g = (g.set_axis_labels(x, y).set_title(title_text))
	plt.show()
	return

def correlation_heatmap(df, title_text):
	corrmat = df.corr()
	f, ax = plt.subplots(figsize=(12, 9))
	sns.heatmap(corrmat, vmax=.8, square=True).set_title(title_text)
	plt.show()
	return

def basic_boxplot(df, colname, title_text):
    sns.boxplot(y=df[colname]).set_title(title_text)
    plt.show()
    return

## Part III: Pre-processing data

def show_nulls(df):
	return df.isna().sum().sort_values(ascending=False)

def fill_whole_df_with_mean(df):
    num_cols = len(df.columns)
    for i in range(0, num_cols):
        df.iloc[:,i] = fill_col_with_mean(df.iloc[:,i])
    return

def fill_col_with_mean(df):
	return df.fillna(df.mean())

#def identify_uni_outliers(df):
    # identify univariate outliers
    # assume normal distribution

# generating features

def generate_dummy(df, colname, attach = False):
    # generate dummy variables from a categorical variable
    # if attach == True, then attach the dummy variables to the original dataframe
    if (attach == False):
        return pd.get_dummies(df[colname])
    else:
        return pd.concat([df, pd.get_dummies(df[colname])], axis = 1)

def discret_eqlbins(df, colname, bin_num):
    # cut continuous variable into bin_num bins
    return pd.cut(df[colname], bin_num)

def discret_quantiles(df, colname, quantile_num):
    # cut cont. variable into quantiles
    return pd.qcut(df[colname], quantile_num)

# feature-scaling

from sklearn import preprocessing
#min_max_scaler = preprocessing.MinMaxScaler()
#df_scaled = min_max_scaler.fit_transform(df)

# standardize data

# scaled_column = scale(df[['x','y']])
from sklearn.preprocessing import scale

def scale_df(df, features_list):
    temp_scaled = scale(df[features_list])
    #return a DF
    return pd.DataFrame(temp_scaled, columns= df.columns)

# split data into training and test sets
from sklearn.model_selection import train_test_split
def split_traintest(df_features, df_target, test_size = 0.2):
    X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_target, test_size = test_size)
    return X_train, X_test, Y_train, Y_test

# methods for training classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def fit_randomforest(x_train, y_train, feature_number, num_trees, depth_num, criterion_choice):
    rf_clf = rf_clf = RandomForestClassifier(max_features = feature_number, n_estimators = num_trees, max_depth = depth_num, criterion = criterion_choice)
    rf_clf.fit(x_train,y_train)
    return rf_clf

def fit_svm(x_train, y_train, c_value, kern, rbf_gam):
    svm_clf = SVC(C = c_value, kernel = kern, gamma = rbf_gam, probability = True)
    svm_clf.fit(x_train, y_train)
    return svm_clf

def fit_naivebayes(x_train, y_train, alpha_value):
    nb_clf = MultinomialNB(alpha = alpha_value)
    nb_clf.fit(x_train,y_train)
    return nb_clf

def fit_knn(x_train, y_train, neighbor_num, distance_type, weight_type):
    knn_clf = KNeighborsClassifier(n_neighbors= neighbor_num, metric= distance_type, weights = weight_type)
    knn_clf.fit(x_train, y_train)
    return knn_clf

def fit_dtree(x_train, y_train, crit_par, split_par, maxdepth_par, minsplit_par,maxfeat_par, minleaf_par, maxleaf_par):
    dt_clf = DecisionTreeClassifier(criterion = crit_par, splitter = split_par, max_depth = maxdepth_par, min_samples_split = minsplit_par, max_features = maxfeat_par, min_samples_leaf = minleaf_par, max_leaf_nodes = maxleaf_par)
    dt_clf.fit(x_train, y_train)
    return dt_clf

def fit_logit(x_train, y_train, penalty_para, c_para):
    logit_clf = LogisticRegression(penalty = penalty_para, C = c_para)
    logit_clf.fit(x_train,y_train)
    return logit_clf

# grid methods

from sklearn.model_selection import GridSearchCV
def grid_cv(clf, param_grid, scoring, cv, x_train, y_train):
    # initialize the grid, scoring = a scoring metric or a dictionary of metrics, 
    # refit is necessary when u have a list of scoring metrics, it determines how the gridsearch algorithm decides the best estimator.
    grid = GridSearchCV(clf(), param_grid, scoring = scoring, cv= cv)
    grid.fit(x_train, y_train)

    # call the best classifier:
    grid.best_estimator_

    # see all performances:
    return grid

def grid_cv_mtp(clf, param_grid, scoring, cv = 5, refit_metric = 'roc'):
    # initialize the grid, scoring = a scoring metric or a dictionary of metrics, 
    # refit is necessary when u have a list of scoring metrics, it determines how the gridsearch algorithm decides the best estimator.
    grid = GridSearchCV(clf(), param_grid, scoring = scoring, cv= cv_num, refit = refit_metric)
    grid.fit(x_train, y_train)

    # call the best classifier:
    grid.best_estimator_

    # see all performances:
    return grid

## Part VI: Evaluating the classifier


#generate predictions according to a custom threshold
def make_predictions(clf, x_test, threshold = 0.7):
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

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

def evaluateAccuracy(clf,predictDF, truthDF):
    correct_pred = 0
    pred_x = clf.predict(predictDF)
    for i in range(0,len(predictDF)):
        if pred_x[i] == truthDF.iloc[i]:
            correct_pred +=1
    return (correct_pred/len(predictDF))

# temporal validation
from dateutil import parser
def create_datetime(df, colname):
    # creates a new column with datetimem objects
    return df[colname].apply(parser.parse)

# a procedure for temporal validation:
# train data  by year. test data on a subset 
#def temp_valididation_year(df, date_column):





# first, some basics of datetime objects:
# for more, here: https://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior

