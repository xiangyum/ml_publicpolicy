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

## Part III: Pre-processing data

def show_nulls(df):
	return df.isna().sum().sort_values(ascending=False)

def fill_na_with_mean(df):
	return df.fillna(df.mean())

## Part IV: Generating features/predictors

# use pd.cut to cut continuous variables into ordered factors. similar to R's cut function.

def dummy_and_merge(df, x):
	dummy = pd.get_dummies(df[x])
	return pd.concat([df, dummy],axis = 1)

## Part V: Building classifier

from sklearn.linear_model import LogisticRegression











