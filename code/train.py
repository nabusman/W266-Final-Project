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
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from tqdm import tqdm

#-- Get Data --#
data_path = '../data'
model_path = '../model_data'

with open(os.path.join(data_path,'full_data/vocabulary_all.pickle'), 'rb') as pickle_file:
  vocab = pickle.load(pickle_file)

x_train = np.load(os.path.join(data_path, 'x_train.npy'))
y_train = np.load(os.path.join(data_path, 'y_train.npy'))
x_val = np.load(os.path.join(data_path, 'x_val.npy'))
y_val = np.load(os.path.join(data_path, 'y_val.npy'))
x_test = np.load(os.path.join(data_path, 'x_test.npy'))
y_test = np.load(os.path.join(data_path, 'y_test.npy'))

max_len = x_train.shape[1]

#-- Build Model --#
batch_size = 32
epochs = 100
num_conv_layers = [1,2,3]
filter_sizes = [128,64,32]
kernel_sizes = [5,4,3,2,1]
embedding_sizes = [500,750,1000]
activations = ['relu', 'leakyrelu']

#-- Train Model --#
parameter_list = []
for conv_layers in num_conv_layers:
  for filter_size in filter_sizes:
    for kernel_size in kernel_sizes:
      for embedding_size in embedding_sizes:
        for activation in activations:
          parameter_list.append({'conv_layers': conv_layers, 'filter_size': filter_size, \
          'kernel_size':kernel_size, 'embedding_size':embedding_size, 'activation':activation})



for parameters in parameter_list:
  conv_layers = parameters['conv_layers']
  filter_size = parameters['filter_size']
  kernel_size = parameters['kernel_size']
  embedding_size = parameters['embedding_size']
  activation = parameters['activation']
  model_name = 'sentiment_model_' + str(conv_layers) + '_' + str(filter_size) + '_' \
    + str(kernel_size) + '_' + str(embedding_size) + '_' + str(activation) + '.h5'
  if os.path.exists(os.path.join(model_path,model_name)):
    print 'Model exists.. skipping... ' + model_name
    continue
  print 'Starting training on ' + model_name
  model = Sequential()
  model.add(Embedding(len(vocab), embedding_size, input_length = max_len))
  model.add(Dropout(0.5))
  for _ in range(conv_layers):
    model.add(Conv1D(filter_size, kernel_size, padding = 'valid'))
    model.add(Activation('relu')) if activation == 'relu' else model.add(LeakyReLU())
  model.add(GlobalMaxPooling1D())
  model.add(Dense(150))
  model.add(Dropout(0.5))
  model.add(Activation('relu'))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))
  model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
  model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs,
    validation_data = (x_val, y_val), callbacks = [EarlyStopping(patience = 5)])
  if os.path.exists(os.path.join(model_path,model_name)):
    print 'Model exists.. skipping... ' + model_name
    continue
  model.save(os.path.join(model_path,model_name))
  del model

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
