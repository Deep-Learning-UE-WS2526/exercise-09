import pandas as pd
import numpy as np

# read in CSV file
data = pd.read_csv("data/gmb.csv",encoding = 'latin1')

# the first column of the file contains the sentence number
# -- but only for the first token of each sentence.
# The following line fills the rows downwards.
data = data.fillna(method = 'ffill')

# create a list of unique words and assign an integer number to it
# shift by +1 so that 0 can be used as the padding index
unique_words, coded_words = np.unique(data["Word"], return_inverse=True)
data["Word_idx"] = coded_words + 1
EMPTY_WORD_IDX = 0
np.array(unique_words.tolist().append("_____"))
num_words = len(unique_words) + 1

# create a list of unique tags and assign an integer number to it
unique_tags, coded_tags = np.unique(data["Tag"], return_inverse=True)
data["Tag_idx"]  = coded_tags
NO_TAG_IDX = unique_tags.tolist().index("O")
num_words_tag = len(unique_tags)

# for verification and inspection, we can inspect the table so far
# data[1:20]

# We are interested in sentence-wise processing.
# Therefore, we use a function that gives us individual sentences.
def get_sentences(data):
  n_sent=1
  agg_func = lambda s:[(w,p,t)
    for w,p,t in zip(
      s["Word_idx"].values.tolist(),
      s["POS"].values.tolist(),
      s["Tag_idx"].values.tolist())]
  grouped = data.groupby("Sentence #").apply(agg_func)
  return [s for s in grouped]

sentences = get_sentences(data)

from keras.utils import pad_sequences
from keras.utils import to_categorical

# find the maximum length for the sentences
max_len = max([len(s) for s in sentences])

# extract the word index
x = [[w[0] for w in s] for s in sentences]
# extract the tag index
y = [[w[2] for w in s] for s in sentences]

# Pad to a fixed length (post-padding) so LSTM can batch sequences
x = pad_sequences(maxlen=max_len, sequences=x, padding='post', value=EMPTY_WORD_IDX)
y = pad_sequences(maxlen=max_len, sequences=y, padding='post', value=NO_TAG_IDX)

# Keep tag indices as integers (required for CRF or sparse losses)
y = np.array(y)

# split the data into training and test data
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state=1)

# Improved sample_weight calculation
# Use x_train to mask padding so real "O" tokens keep weight
y_train_tags = y_train
pad_mask = (x_train != EMPTY_WORD_IDX)

# Only use non-padding tokens to compute class weights
flat_tags = y_train_tags[pad_mask]
unique_classes = np.unique(flat_tags)
class_weights = compute_class_weight('balanced', classes=unique_classes, y=flat_tags)

# Build a full weight vector for all classes
weights_arr = np.ones(num_words_tag, dtype=np.float32)
weights_arr[unique_classes] = class_weights

# Assign weights per token, but zero out padding
sample_weight = np.where(pad_mask, weights_arr[y_train_tags], 0.0)

# Build sample_weight for test set as well (for evaluation and reporting)
y_test_tags = y_test
pad_mask_test = (x_test != EMPTY_WORD_IDX)
sample_weight_test = np.where(pad_mask_test, weights_arr[y_test_tags], 0.0)

from tensorflow.keras import models, layers, optimizers

model = models.Sequential()
model.add(layers.Input(shape = (max_len,)))
model.add(layers.Embedding(input_dim = num_words, output_dim = 50, mask_zero=True))
model.add(layers.LSTM(units = 128, return_sequences = True))  # Increased units
model.add(layers.Dense(num_words_tag, activation = 'softmax'))
model.summary()

# We use a different optimizer this time
model.compile(optimizer='Adam',
  loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#Pass to model.fit()
history = model.fit(
  x_train, np.array(y_train),
    batch_size=64,
    epochs=10,  # Train for more epochs
    verbose=1,
    sample_weight=sample_weight
  )

model.evaluate(x_test, np.array(y_test), sample_weight=sample_weight_test)

from sklearn.metrics import classification_report

Y_test = y_test
y_pred = np.argmax(model.predict(x_test), axis=2)

# Report only on non-padding tokens to avoid padded inflation.
mask_flat = pad_mask_test.flatten()
print(classification_report(
  Y_test.flatten()[mask_flat],
  y_pred.flatten()[mask_flat],
  zero_division=0,
  target_names=unique_tags
))