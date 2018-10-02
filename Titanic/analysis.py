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
print(data_raw.info())
print(data_raw.sample(10))
