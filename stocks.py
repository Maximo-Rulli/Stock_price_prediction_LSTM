#Importing essential libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from itertools import count
import matplotlib.animation as animation

#Importing the dataset from Amazon stocks over 6150 days
df = pd.read_csv('Amazon.csv')

#LIMITE
START = 5730
LIM = START+6

#Getting the closing values of each day
X = df.Close.values
mean = np.mean(X)
std = np.std(X)

#Normalizing the data
X_norm = (X-mean)/std

#Divide my data into a sliding window of 5 days
X_data = np.zeros((len(X_norm)-6, 5))
Y_data = np.zeros((len(X_norm)-6))
for i in range(len(X_norm)-6):
  X_data[i,0:5] = X_norm[i:i+5]
  Y_data[i] = X_norm[i+6]

#
X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], 1)

#Load the pretrained model
model = keras.models.load_model('best_weights.h5')

#Denormalize the data to plot
Y_pred = (model(X_data)*std)+mean
Y_data = (Y_data*std)+mean

#Create the plot and counter to keep track of the number of days when plotting
fig, ax = plt.subplots()
x_plot = []
ax.plot([], [])
fak_counter = count(START, 1)
counter = count(0, 1)

#Define the function to update the graph to make an animation
def update(i):
  idx = next(counter)
  ran = next(fak_counter)
  if ran<LIM:
    x = np.linspace(START, ran+1, idx+1)
    y_pred = Y_pred[START:ran+1, 0]
    y_data = Y_data[START:ran+1]
    x_plot = x
    plt.cla()
    ax.plot(x_plot, y_pred, color='orange', label='Predicted')
    ax.plot(x_plot, y_data, color='blue', label='Original')

#Define the animation
ani = animation.FuncAnimation(fig=fig, func=update, interval=0, cache_frame_data=False)
plt.show()