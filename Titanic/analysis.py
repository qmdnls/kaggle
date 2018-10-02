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

# Previewing our data
print("Data preview: \n")
print(data1.info())
print("-"*20)
print("Sample data:")
print(data1.sample(10))
print("-"*20)
print('Train columns with null values:\n', data1.isnull().sum())
print("-"*20)
print('Validation columns with null values:\n', data_val.isnull().sum())
print("-"*20)
print("Data description:")
print(data1.describe(include = 'all'))

# Now let's clean both our datasets (train and validation) at once
print("\nCleaning data...")
for dataset in data_cleaner:
	# Complete missing age values with the median
	dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

	# Complete missing embarked values with the mode
	dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

	# Complete missing fare values with the median
	dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

# Drop the incomplete cabin column, also exclude PassengerId and Ticket (random unique identifiers)
print("Exclude incomplete and random columns...")
drop_columns = ['Cabin', 'PassengerId', 'Ticket']
data1.drop(drop_columns, axis=1, inplace = True)

# Feature engineering for our train and validations data sets
print("Feature engineering...")
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

print("Done")
print("\n")

# Show count for titles (we might have to clean them up a bit)
print("Title value counts:\n")
print(data1['Title'].value_counts())

# Let's clean up some of those rare title names
print("\n")
print("Cleaning up titles...")
print("\n")
stat_min = 10 # common minimum sample size in statistics
title_names = (data1['Title'].value_counts() < stat_min)
data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

# And finally print the final title counts
print("Title value counts:\n")
print(data1['Title'].value_counts())



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
data1_x_calc = ['Sex_code', 'Pclass', 'Embarked_Code', 'Title_Code', 'SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] # For algorithmic calculation
data1_xy = target + data1_x

# Define our features w/ bins (removes continuous variables)
data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = target + data1_x_bin

# Define features and target variables for dummy features original
data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = target + data1_x_dummy

print('Train columns with null values: \n', data1.isnull().sum())
print("-"*10)
print (data1.info())
print("-"*10)

print('Test/Validation columns with null values: \n', data_val.isnull().sum())
print("-"*10)
print (data_val.info())
print("-"*10)

