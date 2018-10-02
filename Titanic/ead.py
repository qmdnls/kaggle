#!/usr/bin/env python3

# Solution based on the following kernel:
# https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy

# We will use scikit-learn to develop our machine learning algorithms
# For data visualization we will use seaborn and matplotlib


# Load important python libraries for data analysis
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

import xgboost
print("xgboost version: {}". format(xgboost.__version__))

# Misc libraries
import random
import time


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*30)


# Input data files are available in the "input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Importing common modeling algorithms from scikit-learn and xgboost
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

# Importing common model helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

# Visualization using matplotlib and seaborn
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

# Configure some visualization defaults
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8


# The dataset is split into training data, test data and validation data
# A training data file has  been provided in the competition which we will split into training and test data later
# Furthermore a file for final validaton (test.csv) has been provided which will be used for final submission in the competition

# Import the training data
data_raw = pd.read_csv('input/train.csv')

# Import the validation data
data_val = pd.read_csv('input/test.csv')

# Let's create a copy of our data
data1 = data_raw.copy(deep = True)

# To clean our data
data_cleaner = [data1, data_val]

# Now let's clean both our datasets (train and validation) at once
print("Cleaning data...\n")
for dataset in data_cleaner:
	# Complete missing age values with the median
	dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

	# Complete missing embarked values with the mode
	dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

	# Complete missing fare values with the median
	dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

# Drop the incomplete cabin column, also exclude PassengerId and Ticket (random unique identifiers)
drop_columns = ['Cabin', 'PassengerId', 'Ticket']
data1.drop(drop_columns, axis=1, inplace = True)

# Feature engineering for our train and validations data sets
for dataset in data_cleaner:
	# Create a discrete variable for family size
	dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
	# Set whether person is alone (no other family aboard)
	dataset['IsAlone'] = 1 # Initialize to yes
	dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # If family size > 1 they are not alone

	# Create a varialbe for the title, split from name
	dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

	# Keep in mind qcut vs cut
	# Create fare bins using qcut for frequency bins
	dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
	# Create age bins using cut for value bins
	dataset['AgeBin'] = pd.cut(dataset['Age'].astype('int'), 4)

# Let's clean up some of those rare title names
stat_min = 10 # common minimum sample size in statistics
title_names = (data1['Title'].value_counts() < stat_min)
data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)


# Convert objects to category using Label Encoder for train and test/validation dataset
label = LabelEncoder()
for dataset in data_cleaner:
	dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
	dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
	dataset['Title_Code'] = label.fit_transform(dataset['Title'])
	dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
	dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

# Define the target variable
target = ['Survived']

# Define our feature variables
data1_x = ['Sex', 'Pclass', 'Embarked', 'Title', 'SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] # Use these for charts (prettier)
data1_x_calc = ['Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code', 'SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] # For algorithmic calculation
data1_xy = target + data1_x

# Define our features w/ bins (removes continuous variables)
data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = target + data1_x_bin

# Define features and target variables for dummy features original
data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = target + data1_x_dummy

# Let's now split the training data 75/25 (to avoid overfitting)
# train_test_split defaults to splitting off .25 of the dataset as a test set
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[target], random_state = 0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[target], random_state = 0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[target], random_state = 0)

# Print correlation of variables with survival
for x in data1_x:
	if data1[x].dtype != 'float64':
		print('Survival Correlation by:', x)
		print(data1[[x, target[0]]].groupby(x, as_index=False).mean())
		print('-'*30, '\n')

print(pd.crosstab(data1['Title'],data1[target[0]]))

# Let's plot some stuff
# Graph quantitative features
plt.figure(figsize=[16,12])

plt.subplot(231)
plt.boxplot(x=data1['Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(232)
plt.boxplot(data1['Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age (Years)')

plt.subplot(233)
plt.boxplot(data1['FamilySize'], showmeans = True, meanline = True)
plt.title('Family Size Boxplot')
plt.ylabel('Family Size (#)')

plt.subplot(234)
plt.hist(x = [data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']], 
         stacked=True, color = ['b','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(235)
plt.hist(x = [data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']], 
         stacked=True, color = ['b','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(236)
plt.hist(x = [data1[data1['Survived']==1]['FamilySize'], data1[data1['Survived']==0]['FamilySize']], 
         stacked=True, color = ['b','r'],label = ['Survived','Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()

# Graph individual features by survival
fig, saxis = plt.subplots(2, 3,figsize=(16,12))

sns.barplot(x = 'Embarked', y = 'Survived', data=data1, ax = saxis[0,0])
sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=data1, ax = saxis[0,1])
sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=data1, ax = saxis[0,2])

sns.pointplot(x = 'FareBin', y = 'Survived',  data=data1, ax = saxis[1,0])
sns.pointplot(x = 'AgeBin', y = 'Survived',  data=data1, ax = saxis[1,1])
sns.pointplot(x = 'FamilySize', y = 'Survived', data=data1, ax = saxis[1,2])

# Ca heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(data1)

# Show plot
plt.show()
