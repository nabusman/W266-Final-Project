import os
import argparse
import cPickle as pickle
from random import sample, seed

import nltk
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from tqdm import tqdm

def train(layer_order, conv_layers, filter_size, kernel_size, embedding_size, activation, model_dir, data_dir, vocab_size = 10000, l2_value = 0.01):
  #-- Get Data --#
  x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
  y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
  x_val = np.load(os.path.join(data_dir, 'x_val.npy'))
  y_val = np.load(os.path.join(data_dir, 'y_val.npy'))

  num_classes = y_train.shape[1]
  max_len = x_train.shape[1]

  #-- Build Model --#
  batch_size = 32
  epochs = 100
  model_name = 'multinomial_500K_layer_order_sentiment_model_' + str(layer_order) + '_' + str(conv_layers) + '_' + str(filter_size) + '_' \
    + str(kernel_size) + '_' + str(embedding_size) + '_' + str(activation) + '.h5'
  if os.path.exists(os.path.join(model_dir,model_name)):
    print 'Model exists.. skipping... ' + model_name
    return

  print 'Starting training on ' + model_name
  model = Sequential()
  model.add(Embedding(vocab_size, embedding_size, input_length = max_len, embeddings_regularizer = None))
  # model.add(Dropout(0.5))
  for _ in range(conv_layers):
    if layer_order == 'conv-bn-relu':
      model.add(Conv1D(filter_size, kernel_size, padding = 'valid', kernel_regularizer = None))
      model.add(BatchNormalization())
      model.add(Activation('relu')) if activation == 'relu' else model.add(LeakyReLU())
    elif layer_order == 'conv-relu-bn':
      model.add(Conv1D(filter_size, kernel_size, padding = 'valid', kernel_regularizer = None))
      model.add(Activation('relu')) if activation == 'relu' else model.add(LeakyReLU())
      model.add(BatchNormalization())
    elif layer_order == 'relu-conv-bn':
      model.add(Activation('relu')) if activation == 'relu' else model.add(LeakyReLU())
      model.add(Conv1D(filter_size, kernel_size, padding = 'valid', kernel_regularizer = None))
      model.add(BatchNormalization())
    elif layer_order == 'relu-bn-conv':
      model.add(Activation('relu')) if activation == 'relu' else model.add(LeakyReLU())
      model.add(BatchNormalization())
      model.add(Conv1D(filter_size, kernel_size, padding = 'valid', kernel_regularizer = None))
    elif layer_order == 'bn-conv-relu':
      model.add(BatchNormalization())
      model.add(Conv1D(filter_size, kernel_size, padding = 'valid', kernel_regularizer = None))
      model.add(Activation('relu')) if activation == 'relu' else model.add(LeakyReLU())
    elif layer_order == 'bn-relu-conv':
      model.add(BatchNormalization())
      model.add(Activation('relu')) if activation == 'relu' else model.add(LeakyReLU())
      model.add(Conv1D(filter_size, kernel_size, padding = 'valid', kernel_regularizer = None))
  model.add(Flatten())
  model.add(Dense(100))
  # model.add(Dropout(0.5))
  model.add(Activation('relu'))
  model.add(Dense(num_classes))
  model.add(Activation('softmax'))
  model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
  #-- Train Model --#
  model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs,
    validation_data = (x_val, y_val), callbacks = [EarlyStopping(patience = 10)])
  if os.path.exists(os.path.join(model_dir,model_name)):
    print 'Model exists.. skipping... ' + model_name
    return

  model.save(os.path.join(model_dir,model_name))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'Train model')
  parser.add_argument('-l', '--conv_layers', dest = 'conv_layers', type = int,
    required = True, help = 'Number of convolutional layers')
  parser.add_argument('-o', '--layer_order', dest = 'layer_order',
    required = True, help = 'Layer order, permutation of conv, bn, relu seperated by hyphens e.g. conv-bn-relu')
  parser.add_argument('-f', '--filter_size', dest = 'filter_size', type = int,
    required = True, help = 'Size of the filter of the convolutional layers')
  parser.add_argument('-k', '--kernel_size', dest = 'kernel_size', type = int,
    required = True, help = 'Size of the kernel of the convolutional layers')
  parser.add_argument('-e', '--embedding_size', dest = 'embedding_size', type = int,
    required = True, help = 'Embedding layer size')
  parser.add_argument('-a', '--activation', dest = 'activation', choices = ['relu', 'leakyrelu'],
    required = False, default = 'relu', help = 'Activation type, one of `relu` or `leakyrelu`')
  parser.add_argument('-d', '--data_dir', dest = 'data_dir',
    required = True, help = 'Path to the data directory')
  parser.add_argument('-m', '--model_dir', dest = 'model_dir',
    required = True, help = 'Path to the model directory')
  args = parser.parse_args()
  train(**{k:v for k,v in args._get_kwargs()})