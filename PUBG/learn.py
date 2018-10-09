#!/usr/bin/env python3

# Import fast.ai library
from fastai.imports import *
from fastai.structured import *

# Import our models and metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
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
df_trn, y_trn, nas = proc_df(df_raw, 'winPlacePerc')
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(df_trn, y_trn)
set_rf_samples(20000)

# Train model
print("Training...")
m = RandomForestRegressor(n_jobs=-1)
m.fit(X_train, y_train)
print(m.score(X_train, y_train))
print(m.score(X_valid, y_valid))

preds = np.stack([t.predict(X_valid) for t in m.estimators_])
print(preds[:,0], np.mean(preds[:,0]), y_valid[0])
plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)])
plt.show()
