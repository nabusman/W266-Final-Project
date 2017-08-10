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
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from tqdm import tqdm

def train(conv_layers, filter_size, kernel_size, embedding_size, activation, model_dir, data_dir):
  #-- Get Data --#
  with open(os.path.join(data_dir,'full_data/vocabulary_all.pickle'), 'rb') as pickle_file:
    vocab = pickle.load(pickle_file)

  x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
  y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
  x_val = np.load(os.path.join(data_dir, 'x_val.npy'))
  y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
  x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
  y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

  max_len = x_train.shape[1]

  #-- Build Model --#
  batch_size = 32
  epochs = 100
  model_name = 'deep_sentiment_model_' + str(conv_layers) + '_' + str(filter_size) + '_' \
    + str(kernel_size) + '_' + str(embedding_size) + '_' + str(activation) + '.h5'
  if os.path.exists(os.path.join(model_dir,model_name)):
    print 'Model exists.. skipping... ' + model_name
    return

  print 'Starting training on ' + model_name
  model = Sequential()
  model.add(Embedding(len(vocab), embedding_size, input_length = max_len))
  model.add(Dropout(0.5))
  for _ in range(conv_layers):
    model.add(Conv1D(filter_size, kernel_size, padding = 'valid'))
    model.add(Activation('relu')) if activation == 'relu' else model.add(LeakyReLU())
    model.add(MaxPooling1D())

  model.add(Flatten())
  model.add(Dense(150))
  model.add(Dropout(0.5))
  model.add(Activation('relu'))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))
  model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
  #-- Train Model --#
  model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs,
    validation_data = (x_val, y_val), callbacks = [EarlyStopping(patience = 5)])
  if os.path.exists(os.path.join(model_dir,model_name)):
    print 'Model exists.. skipping... ' + model_name
    return

  model.save(os.path.join(model_dir,model_name))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'Train model')
  parser.add_argument('-l', '--conv_layers', dest = 'conv_layers', type = int,
    required = True, help = 'Number of convolutional layers')
  parser.add_argument('-f', '--filter_size', dest = 'filter_size', type = int,
    required = True, help = 'Size of the filter of the convolutional layers')
  parser.add_argument('-k', '--kernel_size', dest = 'kernel_size', type = int,
    required = True, help = 'Size of the kernel of the convolutional layers')
  parser.add_argument('-e', '--embedding_size', dest = 'embedding_size', type = int,
    required = True, help = 'Embedding layer size')
  parser.add_argument('-a', '--activation', dest = 'activation', choices = ['relu', 'leakyrelu'],
    required = True, help = 'Activation type, one of `relu` or `leakyrelu`')
  parser.add_argument('-d', '--data_dir', dest = 'data_dir',
    required = True, help = 'Path to the data directory')
  parser.add_argument('-m', '--model_dir', dest = 'model_dir',
    required = True, help = 'Path to the model directory')
  args = parser.parse_args()
  train(**{k:v for k,v in args._get_kwargs()})