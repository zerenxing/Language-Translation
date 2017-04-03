
import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
#from utils import *
#from rnn_theano import RNNTheano

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

def getData(file):
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print "Reading CSV file..."
    sentences = []
    with open(file, 'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        # Split full comments into sentences
        #sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        for x in reader:
            if len(x) > 0:
                c = x[0].decode('utf-8').lower()
                sentences.append(c)
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s" % (x, sentence_end_token) for x in sentences]
    print "Parsed %d sentences." % (len(sentences))

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique words tokens." % len(word_freq.items())

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    print "Using vocabulary size %d." % vocabulary_size
    print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    print "\nExample sentence: '%s'" % sentences[0]
    print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]

    # Create the training data
    training_data = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_sentences])

    return training_data


def shrink_file(no_of_lines=5000, input=None, output=None):
    i = 0
    with open(input) as f:
        with open(output, "w") as f1:
            for line in f:
                if i < no_of_lines:
                    f1.write(line.replace(","," "))
                    i += 1
                else:
                    return


def get_training_set():

    train_file_en = 'europarl-en-small'
    train_X = getData(train_file_en)

    #train_file_de = 'train_x_de_small.txt'
    #train_Y = getData(train_file_de)

    print ('Done tokenization!')

get_training_set()


#shrink_file(5000, 'europarl-v7.de-en.en', 'europarl-en-small')
