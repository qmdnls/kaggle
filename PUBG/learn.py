#!/usr/bin/env python3

# Import fast.ai library
from fastai.imports import *
from fastai.structured import *

# Import our models and metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
from sklearn import metrics
from os import makedirs

# Define path of our dataset and load files
PATH = "data/"
df_raw = pd.read_csv(f'{PATH}train.csv')
df_test = pd.read_csv(f'{PATH}test.csv')
#os.makedirs('tmp', exist_ok=True)
#df_raw.to_feather('tmp/pubg-raw')
#df_raw = pd.read_feather('tmp/pubg-raw')

# Split dataset
print("Splitting dataset...")
df_trn, y_trn, nas = proc_df(df_raw, 'winPlacePerc')
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(df_trn, y_trn)
set_rf_samples(20000)

# Train model
print("Training...")
m = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, max_features='sqrt', n_jobs=-1)
m.fit(X_train, y_train)
print(m.score(X_train, y_train))
print(m.score(X_valid, y_valid))

# Use model to make our predictions
print("Predicting...")
df_test['winPlacePerc'] = m.predict(df_test)
submit = df_test[['Id', 'winPlacePerc']]
submit.to_csv('submit.csv', index=False)
