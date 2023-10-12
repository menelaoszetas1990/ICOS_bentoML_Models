# Anomaly Detection

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import itertools
import pickle
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

WHICH_CELL = 4

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

# plot the day of sample week
week = 15
plot_data3 = np.zeros((5, 3, 5, 7*24*4))
for cell in range(5):
    for category in range(3):
        for device in range(5):
            # plot_data3[cell, category, device, :] = load_data[cell, category, device, week*7*24*4:(week+1)*7*24*4]
            plot_data3[cell, category, device, :] = anomaly_data[cell, category, device, week*7*24*4:(week + 1)*7*24*4]

plot_data3 = np.sum(np.sum(plot_data3, axis=2), axis=1)
time = np.linspace(0, 167.75, 672)
fig, axs = plt.subplots(5, 1)
labels = []
plt.setp(axs, xticks=time[0:672:12*4].tolist(),
         xticklabels=list(itertools.chain.from_iterable([[str(i*12)+':00' for i in range(2)] for i in range(7)])))
for cell in range(5):
    axs[cell].plot(time, plot_data3[cell, :])
    axs[cell].set_title('cell '+str(cell))

for ax in axs.flat:
    ax.set(xlabel='Time', ylabel='Load')

plt.show()

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
clf = DecisionTreeClassifier(class_weight='balanced')
clf.fit(load_set_scaled, np.reshape(anomaly_set, -1))
print(clf.score(load_set_scaled, anomaly_set))
print(confusion_matrix(anomaly_set, clf.predict(load_set_scaled)))

# Create models folder

if not os.path.isdir('models'):
    os.makedirs('models')

# save
with open('./models/AnomalyDetector_cell' + str(WHICH_CELL) + '.pkl', 'wb') as f:
    pickle.dump(clf, f)

# load
with open('./models/AnomalyDetector_cell' + str(WHICH_CELL) + '.pkl', 'rb') as f:
    clf = pickle.load(f)
