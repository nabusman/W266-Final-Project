import os
import shutil
import json
import cPickle as pickle
from string import punctuation
from random import sample, seed

import nltk
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing import sequence
from tqdm import tqdm


data_path = '../data'
seed(101)
reviews_path = os.path.join(data_path,'yelp_academic_dataset_review.json')
class_map = {'pos' : 1, 'neg' : 0}

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
      ratings.append(class_map['pos'])
    elif stars < 3:
      ratings.append(class_map['neg'])
    else:
      continue
    review_text = review_json['text']
    # Remove punctuation
    review_text = ''.join([x for x in review_text if x not in punctuation])
    # Tokenize and stem words and append to data
    review_text = ' '.join([stemmer.stem(x) for x in nltk.word_tokenize(review_text)])
    reviews.append(review_text)

count_vect = CountVectorizer(stop_words = 'english', max_features = 10000)
train_counts = count_vect.fit_transform(reviews)
vocab = count_vect.get_feature_names()

with open(os.path.join(data_path, 'full_data', 'reviews_all.pickle'), 'wb') as pickle_file:
  pickle.dump(reviews, pickle_file, pickle.HIGHEST_PROTOCOL)

with open(os.path.join(data_path, 'full_data', 'ratings_all.pickle'), 'wb') as pickle_file:
  pickle.dump(ratings, pickle_file, pickle.HIGHEST_PROTOCOL)

with open(os.path.join(data_path, 'full_data', 'vocabulary_all.pickle'), 'wb') as pickle_file:
  pickle.dump(vocab, pickle_file, pickle.HIGHEST_PROTOCOL)

max_len = max([len(x) for x in reviews])
reviews = reviews[:100000] # Take the first 100K reviews

# Separate train, val, test reviews
# Pick the ratio so that it equals a total of 1
ratios = {'train' : .8, 'val' : .1, 'test' : .1}
train_index = np.array(sample(xrange(len(reviews)), int(len(reviews) * ratios['train'])))
val_index = np.array(sample(xrange(len(reviews)), int(len(reviews) * ratios['val'])))
test_index = np.array(sample(xrange(len(reviews)), int(len(reviews) * ratios['test'])))

# initialize empty arrays
x_train = np.empty(len(train_index), dtype = np.object)
x_val = np.empty(len(val_index), dtype = np.object)
x_test = np.empty(len(test_index), dtype = np.object)
y_train = np.array([ratings[x] for x in train_index])
y_val = np.array([ratings[x] for x in val_index])
y_test = np.array([ratings[x] for x in test_index])

for idx,rev_idx in enumerate(tqdm(train_index)):
  review = nltk.word_tokenize(reviews[rev_idx])
  x_train[idx] = [vocab.index(w) for w in review if w in vocab]

for idx,rev_idx in enumerate(tqdm(val_index)):
  review = nltk.word_tokenize(reviews[rev_idx])
  x_val[idx] = [vocab.index(w) for w in review if w in vocab]

for idx,rev_idx in enumerate(tqdm(test_index)):
  review = nltk.word_tokenize(reviews[rev_idx])
  x_test[idx] = [vocab.index(w) for w in review if w in vocab]

x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_val = sequence.pad_sequences(x_val, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

# Save arrays to disk
np.save(os.path.join(data_path, 'train_index'), train_index)
np.save(os.path.join(data_path, 'val_index'), val_index)
np.save(os.path.join(data_path, 'test_index'), test_index)
np.save(os.path.join(data_path, 'x_train'), x_train)
np.save(os.path.join(data_path, 'y_train'), y_train)
np.save(os.path.join(data_path, 'x_val'), x_val)
np.save(os.path.join(data_path, 'y_val'), y_val)
np.save(os.path.join(data_path, 'x_test'), x_test)
np.save(os.path.join(data_path, 'y_test'), y_test)
