import os
import cPickle as pickle
from random import sample, seed

import nltk
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping
from tqdm import tqdm

#-- Get Data --#
data_path = '../data'

x_train = np.load(os.path.join(data_path, 'x_train.npy'))
y_train = np.load(os.path.join(data_path, 'y_train.npy'))
x_val = np.load(os.path.join(data_path, 'x_val.npy'))
y_val = np.load(os.path.join(data_path, 'y_val.npy'))
x_test = np.load(os.path.join(data_path, 'x_test.npy'))
y_test = np.load(os.path.join(data_path, 'y_test.npy'))

#-- Build Model --#
# Test number of conv layers (1,2,3), size of filter sizes (256,128,64), kernel sizes (5,4,3), embedding size (1000,5000) and leaky relu
model = Sequential()
model.add(Embedding(len(vocab), 1000, input_length = max_len))
model.add(Dropout(0.5))
model.add(Conv1D(256, 5, padding = 'valid', activation = 'relu', name = 'conv1'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1024))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#-- Train Model --#
batch_size = 64
epochs = 100
model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs,
  validation_data = (x_val, y_val), callbacks = [EarlyStopping(patience = 5)])