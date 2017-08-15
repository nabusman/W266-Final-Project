import os
import argparse
import cPickle as pickle

import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score


def predict(model_path, x_test_path, y_test_path, vocab_path, results_dir):
	"""
	Predicts using the model at model_path on the data at x_test_path
	and outputs a csv in results_dir including the corresponding values in y_test_path
	and the text seen by the model using the vocabulary at vocab_path
	"""
	result_file_path = os.path.join(results_dir, model_path.split('/')[-1].split('.')[0] + '_results.csv')
	if os.path.exists(result_file_path):
		print 'Skipping: ' + model_path
		return False
	print 'Predicting: ' + model_path
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)

	x_test = np.load(x_test_path)
	y_test = np.load(y_test_path)
	with open(vocab_path, 'rb') as pickle_file:
		vocab = pickle.load(pickle_file)

	model = load_model(model_path)
	predictions = model.predict(x_test)
	print('###### - ' + str(accuracy_score(y_test, map(lambda x: round(x), predictions))) + ' - ######')
	with open(result_file_path, 'w') as results_file:
		for i in xrange(len(predictions)):
			results_file.write(' '.join([vocab[x] for x in x_test[i] if x != 0]).encode('utf-8').strip() + ',' + str(predictions[i][0]) + ',' + str(y_test[i]) + '\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Predict the test set results')
	parser.add_argument('-m', '--model_path', dest = 'model_path', 
		required = True, help = 'Path to the model')
	parser.add_argument('-y', '--y_test', dest = 'y_test_path', 
		required = True, help = 'Path to the y_test.npy file')
	parser.add_argument('-x', '--x_test', dest = 'x_test_path', 
		required = True, help = 'Path to the x_test.npy file')
	parser.add_argument('-r', '--results_dir', dest = 'results_dir', 
		required = False, default = '../results', help = 'Path to the results directory')
	parser.add_argument('-v', '--vocab_path', dest = 'vocab_path', 
		required = True, help = 'Path to the vocabulary file')
	args = parser.parse_args()
	predict(args.model_path, args.x_test_path, args.y_test_path, args.vocab_path, args.results_dir)