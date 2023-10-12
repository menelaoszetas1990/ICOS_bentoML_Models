# Recurrent Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
from functools import reduce
import bentoml

for _cell in range(5):
    print('cell: ', _cell)
    WHICH_CELL = _cell

    model = bentoml.keras.load_model('icos_nkua_rnn_lstm_cell{}:latest'.format(WHICH_CELL))
    bentoml_model = bentoml.keras.get('icos_nkua_rnn_lstm_cell{}:latest'.format(WHICH_CELL))
    sc = bentoml_model.custom_objects['scaler']

    # Importing the training set
    dataset_train = pd.read_csv('./data/Training_data.csv')
    train_data = dataset_train.iloc[:, :].values

    # Split to cell-, category- and device-specific data
    training_data = np.zeros((5, 3, 5, 16608))
    for cell in range(5):
        for category in range(3):
            for device in range(5):
                # picks row that are of a specific cell, category and device
                rows = reduce(np.intersect1d, (np.where(train_data[:, 1] == cell), np.where(train_data[:, 2] == category),
                                               np.where(train_data[:, 3] == device)))
                training_data[cell, category, device] = train_data[rows, 4]

    # select a cell to train
    #training_data[x][y] --> for cell x, category y, row is device, column is timestamp
    training_set = np.sum(np.sum(training_data[WHICH_CELL, :, :, :], axis=0), axis=0)
    # for a specific shell add the loads for each timestamp of each device and category
    training_set = training_set.reshape(16608, 1) # make it 2D as 16608-by-1

    window_len = 2*7*24*4

    # Getting the real stock price of 2017
    dataset_test = pd.read_csv('./data/Test_data.csv')
    test_data = dataset_test.iloc[:, :].values

    # Split to cell-, category- and device-specific data
    testing_data = np.zeros((5, 3, 5, 672))
    for cell in range(5):
        for category in range(3):
            for device in range(5):
                rows = reduce(np.intersect1d, (np.where(test_data[:, 1] == cell),
                                               np.where(test_data[:, 2] == category),
                                               np.where(test_data[:, 3] == device)))
                testing_data[cell, category, device] = test_data[rows, 4]

    # keep only a cell and sum across devices and categories (total load)
    testing_set = np.sum(np.sum(testing_data[WHICH_CELL, :, :, :], axis=0), axis=0)
    testing_set = testing_set.reshape(672, 1) # make it 2D as 672-by-1

    # Getting the test data and the window_len train data (useful to predict the test data)
    dataset_total = np.vstack((training_set, testing_set))
    inputs = dataset_total[len(dataset_total) - len(testing_set) - window_len:]
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(window_len, window_len+10):
        X_test.append(inputs[i-window_len:i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    np.save("./samples/icos_nkua_rnn_lstm_cell{}".format(WHICH_CELL), X_test)
    predicted_load = model.predict(X_test)
    predicted_load = sc.inverse_transform(predicted_load)
    print(predicted_load[:, :])
