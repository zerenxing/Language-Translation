
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

def getData():
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print "Reading CSV file..."
    with open('data/reddit-comments-2015-08.csv', 'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
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
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])




class RNNNumpy(object):
    def __init__(self, word_dim, hidden_dim, bppt_truncate = 4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bppt_truncate = 4

        #Intitialize all the parameters
        """ U = [hidden_dim * word_dim]
            V = [word_dim * hidden_dim]
            W = [hidden_dim * hidden_dim]
                    """
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))


    def forward_propagation(self, x):
        """
        :param x:
        :return:


        S[t] = U[:,x[t]] * W.S[t-1]

        """
        timesteps = len(x)
        S = np.zeros((timesteps + 1), self.hidden_dim)

        O = np.zeros(timesteps, self.word_dim)

        for t in range(timesteps):
            S[t] = self.U[:,x[t]] + np.dot(self.W, S[t-1])
            O[t] = np.softmax(np.dot(self.V, S[t-1]))

