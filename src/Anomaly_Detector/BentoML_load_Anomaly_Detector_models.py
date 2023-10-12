# Anomaly Detection

# Importing the libraries
import numpy as np
import pandas as pd
import bentoml
from functools import reduce
from sklearn.metrics import confusion_matrix

WHICH_CELL = 0

model = bentoml.sklearn.load_model('icos_nkua_clf_anomaly_detection_cell{}:latest'.format(WHICH_CELL))
bentoml_model = bentoml.sklearn.get('icos_nkua_clf_anomaly_detection_cell{}:latest'.format(WHICH_CELL))
sc = bentoml_model.custom_objects['scaler']

# Importing the training set
dataset_train = pd.read_csv('./data/Training_data.csv')
train_data = dataset_train.iloc[:, :].values

# Split to cell-, category- and device-specific data
anomaly_data = np.zeros((5, 3, 5, 16608))
load_data = np.zeros((5, 3, 5, 16608))
for cell in range(5):
    for category in range(3):
        for device in range(5):
            rows = reduce(np.intersect1d, (np.where(train_data[:, 1] == cell),
                                           np.where(train_data[:, 2] == category),
                                           np.where(train_data[:, 3] == device)))
            # load values
            load_data[cell, category, device] = train_data[rows, 4]
            # anomaly values
            anomaly_data[cell, category, device] = train_data[rows, 5]

# select a cell to train
load_set = np.sum(np.sum(load_data[WHICH_CELL, :, :, :], axis=0), axis=0)
xx = np.sum(np.sum(anomaly_data[WHICH_CELL, :, :, :], axis=0), axis=0)
xx[xx != 0] = 1
# make it 2D as 16608-by-1
load_set = load_set.reshape(16608, 1)
# make it 2D as 16608-by-1
anomaly_set = xx.reshape(16608, 1)

# Feature Scaling
load_set_scaled = sc.fit_transform(load_set)

# Creating a data structure with window_len timesteps and 1 output
X_train = []
y_train = []
window_len = 24*4
for i in range(window_len, 16608):
    X_train.append(load_set_scaled[i-window_len:i, 0])
    y_train.append(anomaly_set[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))

# Decision Tree
model.fit(load_set_scaled, np.reshape(anomaly_set, -1))
print(model.score(load_set_scaled, anomaly_set))
print(confusion_matrix(anomaly_set, model.predict(load_set_scaled)))
