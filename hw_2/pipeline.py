import pandas as pd
import numpy as np
import seaborn as sns


## Part 1: read/load data
## use pd.read_csv(), etc.


## Part 2: explore data

# df.columns to show all columns
# use df.describe() for descriptive statistics

def counts_per_variable(df, x):
	return df.groupby(x).size()

def group_and_describe(df, x):
	return df.groupby(x).describe()

def ctab_percent(df, x, y):
	return pd.crosstab(df.loc[:, x], df.loc[:,y], normalize='index')

def ctab_raw(df, x, y):
	return pd.crosstab(df.loc[:, x], df.loc[:,y])

def basic_hist(df, x):
	sns.distplot(df[x])
	plt.show()
	return

def basic_scatter(df, x, y):
	g = sns.lmplot(x, y, data= df)
	g = (g.set_axis_labels(x, y))
	plt.show()
	return

def correlation_heatmap(df):
	corrmat = df.corr()
	f, ax = plt.subplots(figsize=(12, 9))
	sns.heatmap(corrmat, vmax=.8, square=True)
	plt.show()
	return

def point_graph():


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


# feature-scaling

from sklearn import preprocessing
#min_max_scaler = preprocessing.MinMaxScaler()
#df_scaled = min_max_scaler.fit_transform(df)

# standardize data


# scaled_column = scale(df[['x','y']])
from sklearn.preprocessing import scale

def scale_a_df(df, features_list):
    temp_scaled = scale(df[features_list])
    #return a DF
    return pd.DataFrame(temp_scaled, columns= df.columns)


#finish pre-processing by dropping the true labels (forming the x/y splits)

from sklearn.model_selection import train_test_split

#features = [] # Pick the features you want
#df_features = df[features]
#df_target = df['SeriousDlqin2yrs']
#X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_target, test_size=0.2)

## Part IV: Generating features/predictors

# use pd.cut to cut continuous variables into ordered factors. similar to R's cut function.

def dummy_and_merge(df, x):
	dummy = pd.get_dummies(df[x])
	return pd.concat([df, dummy],axis = 1)

## Part V: Building classifier

#from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split

#split data into testing and training data.


#clf = LogisticRegression()
#clf.fit(x_train,y_train)


## Part VI: Evaluating the classifier

def custom_predictions(clf, threshold = 0.7, x_test = x_test, y_test = y_test):
    for i in range(0,len(x_test)):
        
    clf.predict_proba(x_test)

def evaluateAccuracy(clf,predictDF, truthDF):
    correct_pred = 0
    pred_x = clf.predict(predictDF)
    for i in range(0,len(predictDF)):
        if pred_x[i] == truthDF.iloc[i]:
            correct_pred +=1
    return (correct_pred/len(predictDF))



