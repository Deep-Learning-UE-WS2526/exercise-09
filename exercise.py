import pandas as pd
import numpy as np

# read in CSV file
data = pd.read_csv("data/gmb.csv",encoding = 'latin1')

# the first column of the file contains the sentence number
# -- but only for the first token of each sentence.
# The following line fills the rows downwards.
data = data.ffill()

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

# Calculate tag frequencies for weighting
tag_counts = data['Tag'].value_counts()
print("Tag frequency distribution:")
print(tag_counts)

# Create inverse frequency weights (more weight for rarer tags)
# Use log of inverse frequency to prevent extreme weights
total_tags = len(data)
class_weights = {}
for tag in unique_tags:
    frequency = tag_counts[tag] / total_tags
    # Use log of inverse frequency, with minimum weight of 1.0
    weight = max(1.0, np.log(1.0 / frequency))
    class_weights[tag] = weight

print("\nClass weights (higher = rarer tags):")
for tag, weight in sorted(class_weights.items(), key=lambda x: x[1], reverse=True):
    print(f"{tag}: {weight:.2f}")

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

# shorter sentences are now padded to same length, using (index of) padding symbol
x = pad_sequences(maxlen = max_len, sequences = x,
  padding = 'post', value = EMPTY_WORD_IDX)

# we do the same for the y data
y = pad_sequences(maxlen = max_len, sequences = y,
  padding = 'post', value = NO_TAG_IDX)

# but we also convert the indices to one-hot-encoding
y = np.array([to_categorical(i, num_classes = num_words_tag) for i in  y])

# Create a weight vector based on tag frequency
def create_sample_weights(y_data, unique_tags, class_weights):
    """
    Creates sample weights for named entity recognition based on tag frequency:
    - Rarer tags get higher weights (inverse frequency weighting)
    - Padding positions get weight 0.0
    """
    padding_idx = NO_TAG_IDX  # This is the padding index
    
    weights = []
    
    for sentence in y_data:
        sentence_weights = []
        for token_one_hot in sentence:
            # Get the actual tag index (argmax of one-hot encoding)
            tag_idx = np.argmax(token_one_hot)
            
            if tag_idx == padding_idx:
                # Padding position gets weight 0.0
                sentence_weights.append(0.0)
            else:
                # Get the tag name and its frequency-based weight
                tag_name = unique_tags[tag_idx]
                weight = class_weights[tag_name]
                sentence_weights.append(weight)
        
        weights.append(sentence_weights)
    
    return np.array(weights)

# split the data into training and test data
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state=1)

# Create sample weights for training data
sample_weights_train = create_sample_weights(y_train, unique_tags, class_weights)

# Print sample weight statistics
print(f"\nSample weights shape: {sample_weights_train.shape}")
print(f"Unique weight values: {np.unique(sample_weights_train)}")
weight_counts = {val: np.sum(sample_weights_train == val) for val in np.unique(sample_weights_train)}
print(f"Weight distribution in training data: {weight_counts}")

from keras import models, layers, optimizers

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
    verbose = 1,
    sample_weight = sample_weights_train
)

model.evaluate(x_test, np.array(y_test))

from sklearn.metrics import classification_report

Y_test = np.argmax(y_test, axis=2)

y_pred = np.argmax(model.predict(x_test), axis=2)


print(classification_report(Y_test.flatten(), y_pred.flatten(), zero_division=0, target_names=unique_tags))