#!/usr/bin/env python3

# Import fast.ai library
from fastai.imports import *
from fastai.structured import *

# Import our models and metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from os import makedirs

# Define path of our dataset
PATH = "data/"
df_raw = pd.read_csv(f'{PATH}train.csv')
#os.makedirs('tmp', exist_ok=True)
#df_raw.to_feather('tmp/pubg-raw')
#df_raw = pd.read_feather('tmp/pubg-raw')

# Split dataset
print("Splitting dataset...")


# Train model
print("Training...")
m = RandomForestRegressor(n_jobs=-1)
m.fit(df_raw.drop('winPlacePerc', axis=1), df_raw.winPlacePerc)
print(m.score(df_raw.drop('winPlacePerc', axis=1), df_raw.winPlacePerc))
