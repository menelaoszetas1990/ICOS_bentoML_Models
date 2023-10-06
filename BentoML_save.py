# Recurrent Neural Network
# Importing the libraries
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import bentoml

for _cell in range(5):
    print(_cell)
    WHICH_CELL = _cell

    # Importing the training set
    dataset_train = pd.read_csv('Training_data.csv')
    train_data = dataset_train.iloc[:, :].values

    # Split to cell-, category- and device-specific data
    training_data = np.zeros((5, 3, 5, 16608))
    for cell in range(5):
        for category in range(3):
            for device in range(5):
                rows = reduce(np.intersect1d, (np.where(train_data[:, 1] == cell),
                                               np.where(train_data[:, 2] == category),
                                               np.where(train_data[:, 3] == device)))
                training_data[cell, category, device] = train_data[rows, 4]

    # select a cell to train
    training_set = np.sum(np.sum(training_data[WHICH_CELL, :, :, :], axis=0), axis=0)
    # make it 2D as 16608-by-1
    training_set = training_set.reshape(16608, 1)

    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    regressor = load_model('models/LSTM_cell'+str(WHICH_CELL))

    print('BendoML start saving cell {}'.format(_cell))
    bentoml.keras.save_model('LSTM_cell{}'.format(_cell), regressor, custom_objects={"scaler": sc})
    print('BendoML end saving {}'.format(_cell))
