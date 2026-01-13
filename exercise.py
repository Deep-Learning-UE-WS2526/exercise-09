import pandas as pd
import numpy as np

# read in CSV file
data = pd.read_csv("data/gmb.csv",encoding = 'latin1')

# the first column of the file contains the sentence number
# -- but only for the first token of each sentence.
# The following line fills the rows downwards.
data = data.fillna(method = 'ffill')

# create a list of unique words and assign an integer number to it
unique_words, coded_words = np.unique(data["Word"], return_inverse=True)
data["Word_idx"] = coded_words
EMPTY_WORD_IDX = len(unique_words)
np.array(unique_words.tolist().append("_____"))
num_words = len(unique_words)+1

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
x = np.array([ np.array([ w[0] for w in s ]) for s in sentences ])
# extract the tag index
y = np.array([ np.array([ w[2] for w in s ]) for s in sentences ])

# shorter sentences are now padded to same length, using (index of) padding symbol
x = pad_sequences(maxlen = max_len, sequences = x,
  padding = 'post', value = EMPTY_WORD_IDX)

# we do the same for the y data
y = pad_sequences(maxlen = max_len, sequences = y,
  padding = 'post', value = NO_TAG_IDX)

# but we also convert the indices to one-hot-encoding
y = np.array([to_categorical(i, num_classes = num_words_tag) for i in  y])

# split the data into training and test data
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state=1)

from tensorflow.keras import models, layers, optimizers

model = models.Sequential()
model.add(layers.Input(shape = (max_len,)))
model.add(layers.Embedding(input_dim = num_words, output_dim = 50, input_length = max_len))
model.add(layers.LSTM(units = 5, return_sequences = True))
model.add(layers.Dense(num_words_tag, activation = 'softmax'))
model.summary()

# We use a different optimizer this time
model.compile(optimizer='Adam',
  loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(
    x_train, np.array(y_train),
    batch_size = 64,
    epochs = 1,
    verbose = 1
)

model.evaluate(x_test, np.array(y_test))

from sklearn.metrics import classification_report

Y_test = np.argmax(y_test, axis=2)

y_pred = np.argmax(model.predict(x_test), axis=2)


print(classification_report(Y_test.flatten(), y_pred.flatten(), zero_division=0, target_names=unique_tags))