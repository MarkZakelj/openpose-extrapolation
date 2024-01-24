import xgboost as xgb
import pandas as pd
import torch
from os.path import join
import os
import numpy as np

DATA_VERSION = 'v4'
DATA_DIR = f'data/{DATA_VERSION}'

def read_pt_file(file_name):
    tensor = torch.load(file_name)
    return tensor.numpy()

X_train = read_pt_file(join(DATA_DIR, 'poses_missing_train.pt'))
assert (X_train == -10.0).sum() > 0
Y_train = read_pt_file(join(DATA_DIR, 'poses_train.pt'))
assert (Y_train == -10.0).sum() == 0
                       
X_val = read_pt_file(join(DATA_DIR, 'poses_missing_valid.pt'))
Y_val = read_pt_file(join(DATA_DIR, 'poses_valid.pt'))

X_test = read_pt_file(join(DATA_DIR, 'poses_missing_test.pt'))
Y_test = read_pt_file(join(DATA_DIR, 'poses_test.pt'))

dtrain = xgb.DMatrix(X_train, missing=-10.0)
dval = xgb.DMatrix(X_val, missing=-10.0)
dtest = xgb.DMatrix(X_test, missing=-10.0)

# Model parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': ['rmse']
    # Other parameters as needed
}

models = {}
for target in range(36):
    y_train = Y_train[:, target]
    y_val = Y_val[:, target]

    dtrain.set_label(y_train)
    dval.set_label(y_val)

    evallist = [(dval, 'eval'), (dtrain, 'train')]
    num_round = 100

    print(f"Training model for {target}")
    models[target] = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=10)

# Make predictions on test set
mses = []
for target in range(36):
    print(f"Predicting for {target}")
    error = (Y_test[:, target] - models[target].predict(dtest)) ** 2
    mses.append(error.mean())

models_dir = f'models/xgboost/{DATA_VERSION}'
os.makedirs(models_dir, exist_ok=True)
for mod in range(36):
    models[mod].save_model(join(models_dir, f'model_{mod}.json'))

print(f'MSE: {np.mean(mses)} : variance: {np.var(mses)} std: {np.std(mses)}')
print(mses)