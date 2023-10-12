# Anomaly Detection
# Importing the libraries
import numpy as np
import pandas as pd
import pickle
import bentoml
from functools import reduce
from sklearn.preprocessing import MinMaxScaler

for _cell in range(5):
    print('cell: ', _cell)
    WHICH_CELL = _cell

    # Importing the training set
    dataset_train = pd.read_csv('./data/Training_data.csv')
    train_data = dataset_train.iloc[:, :].values

    # Part 1 - Data Preprocessing
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
    sc = MinMaxScaler(feature_range=(0, 1))
    load_set_scaled = sc.fit_transform(load_set)

    # load
    with open('./models/AnomalyDetector_cell' + str(WHICH_CELL) + '.pkl', 'rb') as f:
        clf = pickle.load(f)

        print('BendoML start saving cell {}'.format(_cell))
        bentoml.sklearn.save_model('icos_nkua_clf_anomaly_detection_cell{}'.format(_cell), clf,
                                   custom_objects={"scaler": sc})
        print('BendoML end saving cell {}'.format(_cell))
