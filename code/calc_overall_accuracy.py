import os
import re

import numpy as np
from sklearn.metrics import accuracy_score


results_dir = '../results'
overall_results_path = '../overall_results.csv'
rounder = np.vectorize(lambda x: int(round(float(x))))
clean_int = np.vectorize(lambda x: int(x.strip()))

results_list = filter(lambda x: '.csv' in x, os.listdir(results_dir))

if not os.path.exists(overall_results_path):
  with open(overall_results_path, 'w') as overall_results:
    overall_results.write('File Path, Convolutional Layers, Filter Sizes, Kernel Sizes, Embedding Size, Accuracy\n')

for result_file in results_list:
  with open(os.path.join(results_dir, result_file)) as result_data:
    result = np.array([x.split(',') for x in result_data.readlines()])
  accuracy = accuracy_score(clean_int(result[:,2]), rounder(result[:,1]))
  print result_file
  print str(accuracy)
  layers, filters, kernels, embedding = map(lambda x: x.replace('_', ''), re.findall(r'(\d+\_)', result_file))
  with open(overall_results_path, 'a') as overall_results:
    overall_results.write(result_file + ',' + layers + ',' + filters + ',' + kernels + ',' + embedding + ',' + str(accuracy) + '\n')