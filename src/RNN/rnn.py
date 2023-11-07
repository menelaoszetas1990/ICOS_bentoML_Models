# Recurrent Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import itertools
from functools import reduce

for _cell in range(5):
    print('cell: ', _cell)
    WHICH_CELL = _cell

    # Importing the training set
    dataset_train = pd.read_csv('./data/Training_data.csv')
    train_data = dataset_train.iloc[:, :].values

    # Split to cell-, category- and device-specific data
    training_data = np.zeros((5, 3, 5, 16608))
    for cell in range(5):
        for category in range(3):
            for device in range(5):
                rows = reduce(np.intersect1d, (np.where(train_data[:, 1] == cell), np.where(train_data[:, 2] == category), np.where(train_data[:, 3] == device)))
                # load values
                training_data[cell, category, device] = train_data[rows, 4]

    # plot training data (averaged every 1 day)
    # total_days = 16608/(4*24)
    # plot_data = np.zeros((5, 3, 5, 4*24))
    # for day in range(int(total_days)):
    #     for cell in range(5):
    #         for category in range(3):
    #             for device in range(5):
    #                 plot_data[cell, category, device, :] += training_data[cell, category, device, day*96:(day+1)*96]

    # plot_data = plot_data/total_days
    # time = np.linspace(0, 23.75, 96)
    # fig, axs = plt.subplots(5, 15)
    # for cell in range(5):
    #     for category in range(3):
    #         for device in range(5):
    #             axs[cell, category*5 + device].plot(time, plot_data[cell, category, device, :])
    #             axs[cell, category*5 + device].set_title('cell '+str(cell)+', category '+str(category)+', device '+str(device))
    #
    # for ax in axs.flat:
    #     ax.set(xlabel='Time', ylabel='Load')

    # plt.show()

    # total load of each cell (averaged evey 1 day)
    # plot_data2 = np.sum(np.sum(plot_data, axis=2), axis=1)
    # time = np.linspace(0, 23.75, 96)
    # fig, axs = plt.subplots(1, 5)
    # for cell in range(5):
    #     axs[cell].plot(time, plot_data2[cell, :])
    #     axs[cell].set_title('cell '+str(cell))
    #
    # for ax in axs.flat:
    #     ax.set(xlabel='Time', ylabel='Load')

    # plt.show()

    # plot the day of sample week
    # week = 15
    # plot_data3 = np.zeros((5, 3, 5, 7*24*4))
    # for cell in range(5):
    #     for category in range(3):
    #         for device in range(5):
    #             plot_data3[cell, category, device, :] = training_data[cell, category, device, week*7*24*4:(week+1)*7*24*4]
    #
    # plot_data3 = np.sum(np.sum(plot_data3, axis=2), axis=1)
    # time = np.linspace(0, 167.75, 672)
    # # time = np.hstack([np.linspace(0, 23.75, 96) for i in range(7)])
    # fig, axs = plt.subplots(5, 1)
    # labels = []
    # plt.setp(axs, xticks=time[0:672:12*4].tolist(), xticklabels=list(itertools.chain.from_iterable([[str(i*12)+':00' for i in range(2)] for i in range(7)])))
    # for cell in range(5):
    #     axs[cell].plot(time, plot_data3[cell, :])
    #     axs[cell].set_title('cell '+str(cell))
    #
    # for ax in axs.flat:
    #     ax.set(xlabel='Time', ylabel='Load')
    #
    # plt.show()

    # select a cell to train
    training_set = np.sum(np.sum(training_data[WHICH_CELL, :, :, :], axis=0), axis=0)
    training_set = training_set.reshape(16608, 1) # make it 2D as 16608-by-1

    # Feature Scaling
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # Creating a data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []
    window_len = 2*7*24*4
    for i in range(window_len, 16608):
        X_train.append(training_set_scaled[i-window_len:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Part 2 - Building the RNN

    # # Initialising the RNN
    # regressor = Sequential()
    #
    # # Adding the first LSTM layer and some Dropout regularisation
    # regressor.add(LSTM(units=50, return_sequences = True, input_shape=(X_train.shape[1], 1)))
    # regressor.add(Dropout(0.2))
    #
    # # Adding a second LSTM layer and some Dropout regularisation
    # regressor.add(LSTM(units=50, return_sequences=True))
    # regressor.add(Dropout(0.2))
    #
    # # Adding a third LSTM layer and some Dropout regularisation
    # regressor.add(LSTM(units=50, return_sequences=True))
    # regressor.add(Dropout(0.2))
    #
    # # Adding a fourth LSTM layer and some Dropout regularisation
    # regressor.add(LSTM(units=50))
    # regressor.add(Dropout(0.2))
    #
    # # Adding the output layer
    # regressor.add(Dense(units=1))
    #
    # # Compiling the RNN
    # regressor.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the Training set
    # regressor.fit(X_train, y_train, epochs=20, batch_size=64)

    # Create models folder
    # import os
    # # if not os.path.isdir('models'):
    # #     os.makedirs('models')
    # if not os.path.isdir('models_anomaly'):
    #     os.makedirs('models_anomaly')

    # save the cell-specific model
    # regressor.save('models/LSTM_cell'+str(WHICH_CELL))
    # regressor.save('models_anomaly/LSTM_cell'+str(WHICH_CELL))

    # load a saved model
    from tensorflow import keras
    regressor = keras.models.load_model('models/LSTM_cell'+str(WHICH_CELL))

    # Part 3 - Making the predictions and visualising the results

    # Getting the real stock price of 2017
    dataset_test = pd.read_csv('./data/Test_data.csv')
    test_data = dataset_test.iloc[:, :].values

    # Split to cell-, category- and device-specific data
    testing_data = np.zeros((5, 3, 5, 672))
    for cell in range(5):
        for category in range(3):
            for device in range(5):
                rows = reduce(np.intersect1d, (np.where(test_data[:, 1] == cell), np.where(test_data[:, 2] == category), np.where(test_data[:, 3] == device)))
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
    for i in range(window_len, 2016):
        X_test.append(inputs[i-window_len:i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_load = regressor.predict(X_test)
    predicted_load = sc.inverse_transform(predicted_load)

    np.save("./samples/icos_nkua_rnn_lstm_cell{}_groundtruth_values".format(WHICH_CELL), testing_set[:10, :])

    # Visualising the results
    # plt.plot(testing_set, color='red', label='Real load')
    # plt.plot(predicted_load, color='blue', label='Predicted load')
    # plt.title('Load Prediction of cell ' + str(WHICH_CELL))
    # plt.xlabel('Time')
    # plt.ylabel('Load')
    # plt.legend()
    # plt.show()
