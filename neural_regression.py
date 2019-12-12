"""Import necessary libraries"""

from __future__ import absolute_import, division, print_function, unicode_literals
from mpl_toolkits.mplot3d import Axes3D  
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import losses


import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

#####################################################################################################

"""Read the data"""
column_names = ['Width', 'Color', 'Steering_angle_difference']
raw_dataset = pd.read_csv('data/new_data.csv', sep = ',', skipinitialspace=True, dtype = float)
dataset = raw_dataset.copy()


train_dataset = dataset.sample(frac = 0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

#####################################################################################################

"""Get the statistics of the data"""
train_stats = train_dataset.describe()
train_stats.pop("Steering_angle_difference")
train_stats =  train_stats.transpose()


train_labels =train_dataset.pop('Steering_angle_difference')
test_labels = test_dataset.pop('Steering_angle_difference')

#######################################################################################################

"""Normalize the data"""
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
    
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

########################################################################################################

"""Build the model"""
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation = 'relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(0.001)
    
    model.compile(loss = 'mse', optimizer=optimizer, metrics=['mse', 'mae'])

    return model


model = build_model()

print(model.summary())

#########################################################################################################

"""Train it for a lot of epochs"""
EPOCHS = 500

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, verbose=0, validation_split = 0.4,
  callbacks=[tfdocs.modeling.EpochDots()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

plotter.plot({'Basic': history}, metric = "mse")
plt.ylim([0, 10])
plt.ylabel('MSE [Steering difference]')
plt.show()

########################################################################################################

"""Based on the training and validation error, determine the number of epochs to train the model to prevent over or underfitting"""
model = build_model()

EPOCHS = 200
early_history = model.fit(normed_train_data, train_labels, 
                    epochs=EPOCHS, validation_split = 0.4, verbose=0, 
                    )

########################################################################################################

"""Test the model"""
test_predictions = model.predict(normed_test_data).flatten()

########################################################################################################

"""Plot true vs predicted value graph"""
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Steering Angle Difference]')
plt.ylabel('Predictions [Steering Angle Difference]')
lims = [0, 5]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

########################################################################################################

"""Plot test error histogram"""
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [Steering Angle Difference]")
_ = plt.ylabel("Count")
plt.show()

