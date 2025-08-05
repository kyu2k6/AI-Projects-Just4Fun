import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

# gets the file from online
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# read the file and treat all characters as lower case
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

# selecting a part of the text
text = text[300000:800000]

characters = sorted(set(text))

# create a dictionary where character is key and index is value
# will look like {'a': 1, 'c' : 2, ....} and vice versa
char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

# 40 characters to predict next character
SEQ_LENGTH = 40

# how many steps before going to where the next sequence starts
STEP_SIZE = 3

# array of sentences where we try to predict the next character
sentences = []

# array of next characters to those arrays
next_characters = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i : i + SEQ_LENGTH])
    next_characters.append(text[i + SEQ_LENGTH])

# initialize a numpy array with zeros
# dimensions include one with length of all the sentences we have
#                    one with number of positions in each sentence or length of each sentence
#                    one with all the possible characters we can have
# will set to true or 1 if character appears at the position
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool)

# target data or what will be the next character for a sentence using the enumeration index number
# and the sentence that is being looked at
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)

# filling the arrays
for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

# building the neural network
model = Sequential()
# LongShortTermMemory
model.add(Input(shape=(SEQ_LENGTH, len(characters))))
model.add(LSTM(128))
model.add(Dense(len(characters)))
# softmax scales the output so that all values add up to 1
# output will always be a probability of how likely characters will be the next
# character that we are looking for.
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

# batch size is how many examples are we going to put in the network at once
# epochs is how many times the network will run the same data again
model.fit(x, y, batch_size=256, epochs=4)

model.save('textgenerator.keras')

model = tf.keras.models.load_model('textgenerator.keras')

# takes the predictions of our model and picks one character
# high temp means takes a more risky and experimental character
# low temp means takes a more safe character
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):

    # chooses a random start point in your training text
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence

    # loop to generate new characters
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1

        # runs model on the current sentence and returns a probability distribution
        # over all possible characters
        predictions = model.predict(x, verbose=0)[0]

        # uses the sample function to pick next character and converts index back to actual character
        next_index = sample(predictions, temperature)
        next_char = index_to_char[next_index]

        # adds chosen character to generated text and updates the context sentence
        # by removing the first character and adding the new one at the end to keep
        # the sequence length consistent
        generated += next_char
        sentence = sentence[1:] + next_char

    return generated

print('---------------0.2---------------')
print(generate_text(300, temperature=0.2))
print('---------------0.4---------------')
print(generate_text(300, temperature=0.4))
print('---------------0.6---------------')
print(generate_text(300, temperature=0.6))
print('---------------0.8---------------')
print(generate_text(300, temperature=0.8))
print('---------------1.0---------------')
print(generate_text(300, temperature=1.0))