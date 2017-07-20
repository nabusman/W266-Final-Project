import os
import shutil
import json
import cPickle as pickle
from string import punctuation
from random import choice

import nltk
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer


data_path = '../data'
reviews_path = os.path.join(data_path,'yelp_academic_dataset_review.json')
classes = ['pos', 'neg']

reviews = []
ratings = []

# Stem reviews before saving them
stemmer = nltk.stem.porter.PorterStemmer()
with open(reviews_path) as review_file:
  for line in tqdm(review_file):
    # Load review text
    review_json = json.loads(line)
    # Set reviews to positive or negative, skip if ==3
    stars = review_json['stars']
    if stars > 3:
      ratings.append(classes[0])
    elif stars < 3:
      ratings.append(classes[1])
    else:
      continue
    review_text = review_json['text']
    # Remove punctuation
    review_text = ''.join([x for x in review_text if x not in punctuation])
    # Tokenize and stem words and append to data
    review_text = ' '.join([stemmer.stem(x) for x in nltk.word_tokenize(review_text)])
    reviews.append(review_text)

with open(os.path.join(data_path,'reviews.pickle'), 'wb') as pickle_file:
  pickle.dump(reviews, pickle_file, pickle.HIGHEST_PROTOCOL)

with open(os.path.join(data_path,'ratings.pickle'), 'wb') as pickle_file:
  pickle.dump(ratings, pickle_file, pickle.HIGHEST_PROTOCOL)


max_len = max([len(x) for x in reviews])
count_vect = CountVectorizer(stop_words = 'english', max_features = 10000)
train_counts = count_vect.fit_transform(reviews)
vocab = count_vect.get_feature_names()

with open(os.path.join(data_path,'vocabulary.pickle'), 'wb') as pickle_file:
  pickle.dump(vocab, pickle_file, pickle.HIGHEST_PROTOCOL)

train_path = '../data/reviews/train'
val_path = '../data/reviews/val'
test_path = '../data/reviews/test'

[os.makedirs(os.path.join(x,y)) for x in [train_path, val_path, test_path] for y in ['pos', 'neg'] if not os.path.exists(os.path.join(x,y))]

# Train:Validation:Test ratio of 8:1:1
train_val_test_ratio = [train_path] * 8 + [val_path] * 1 + [test_path] * 1
data_destination = [choice(train_val_test_ratio) for _ in xrange(len(reviews))]
file_name = 0

for i in tqdm(xrange(len(reviews))):
  # For every review and ratings create a matrix of W x V (w is # of words in review and V is the size of the vocabulary)
  review = nltk.word_tokenize(reviews[i])
  rating = ratings[i]
  matrix = []
  for j in xrange(max_len):
    row = [0] * len(vocab)
    if j < len(review) and review[j] in vocab:
      row[vocab.index(review[j])] = 1
    matrix.append(row)
  np.save(os.path.join(data_destination[i], rating, 'review_' + str(file_name)), np.array(matrix))
  file_name += 1