
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing import sequence
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import argparse
from keras.models import model_from_json
import cPickle
from keras.models import model_from_json
from keras.models import load_model
import os.path

BASE = os.path.dirname(os.path.abspath(__file__))
def test(sentence):
    json_file = open(os.path.join(BASE, "all_text.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
# load weights into new model
    loaded_model.load_weights(os.path.join(BASE, "model.h5"))
    print("Loaded model from disk")
 
# evaluate loaded model on test data

    optimizer = RMSprop(lr=0.01)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    name = "all_text"
    path = name + ".txt"
    text = open(os.path.join(BASE, path), 'r').read().lower()
#print('corpus length:', len(text))

    chars = sorted(list(set(text)))
#print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
    maxlen = 40
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
#print('nb sequences:', len(sentences))

    partial_length = len(sentences)/3
    sentences = sentences[:partial_length]
#print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    start_index = random.randint(0, len(text) - maxlen - 1)
    generatedList = []
    for diversity in [0.2, 0.5, 1.0]: #1.2s
        print()
        print('----- diversity:', diversity)

        generated = ''
    # sentence = text[start_index: start_index + maxlen]
        if len(sentence) > maxlen:
            sentence = sentence[len(sentence)-maxlen:]
        else:
            sentence = "a"*(maxlen-len(sentence))+sentence

        generated += sentence
    # print('----- Generating with seed: "' + sentence + '"')
    # sys.stdout.write(generated)
        for bal in range(0,1):
            for i in range(1100):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = loaded_model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()
        generatedList.append(generated)
    return generatedList


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration




    #save model
# serialize model to JSON




# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
