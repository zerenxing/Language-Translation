import numpy as np
from test import *
from datetime import datetime
import sys
import os
import time

class VanillaRnn(object):
    
    def __init__(self, source_dim, target_dim, hidden_dim, bptt_truncate=5):
        # Assign instance variables
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.We = np.random.uniform(-np.sqrt(1./source_dim), np.sqrt(1./source_dim), (hidden_dim, source_dim))
        self.Ue = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.Wd = np.random.uniform(-np.sqrt(1./target_dim), np.sqrt(1./target_dim), (hidden_dim, target_dim))
        self.Ud = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.P = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (target_dim,hidden_dim))

    def calculate_loss(self, x, y):
        Tdec = len(y)
        h = self.encoder(x)
        c = h[-1]
        s,o = self.decoder(x, c, y)
        target_probability = o[np.arange(Tdec), y]
        L = -1 *np.sum(np.log(target_probability))
        return L

    def total_loss(self, x, y):
        # Divide the total loss by the number of training examples
        L = 0
        N = 0
        for i in np.arange(len(y)):
            L += self.calculate_loss(x[i],y[i])
            N += len(y[i])
        return L / N

    def encoder(self, x):
        # The total number of time steps
        Tenc = len(x)
        # During forward propagation we save all hidden states in h because need them for back propagation.
        h = np.zeros((Tenc, self.hidden_dim))
        # For each time step...
        for t in np.arange(Tenc):
            # Note that we are indexing W by x[t]. This is the same as multiplying U with a one-hot vector.
            h[t] = activation(self.We[:, x[t]] + np.dot(self.Ue, h[t-1]))
        return h

    def decoder(self, x, c, y=None):
        # Training case
        if (y != None):
            # The total number of time steps
            Tdec = len(y)
            # During forward propagation we save all hidden states in h because need them for back propagation.
            s = np.zeros((Tdec, self.hidden_dim))
            s[0] = activation(np.dot(self.V, c))
            o = np.zeros((Tdec, self.target_dim))
            o[0] = softmax(np.dot(self.P, s[0]))
            # For each time step...
            for t in np.arange(1,Tdec):
                # Note that we are indexing W by x[t]. This is the same as multiplying U with a one-hot vector.
                s[t] = activation(self.Wd[:, y[t - 1]] + np.dot(self.Ud, s[t - 1]) + np.dot(self.V, c))
                o[t] = softmax(np.dot(self.P, s[t]))
            return [s, o]
        # Translation case
        else:
            # Start translation without previous word presented
            s = [activation(np.dot(self.V, c))]
            o = [softmax(np.dot(self.P, s[0]))]
            y = [np.argmax(o)]
            t = 1
            # Keep translation till the ending sign is reached
            while y[-1] != word2vec_target[sentence_end_token] and len(y)<30 :
                s.append(activation(self.Wd[:, y[t - 1]] + np.dot(self.Ud, s[t - 1]) + np.dot(self.V, c)))
                o.append(softmax(np.dot(self.P, s[t])))
                trial = word2vec_target[unknown_token]
                while(trial == word2vec_target[unknown_token]):
                    samples = np.random.multinomial(1, o[t])
                    trial = np.argmax(samples)
                y.append(trial)
                t = t + 1
            y = np.asarray(y)
            o = np.asarray(o)
            s = np.asarray(s)
        return [s, o, y]

    def translate(self, x):
        target_corpus=[]
        for i in range(len(x)):
            h = self.encoder(x[i])
            s, o, y = self.decoder(x[i], h[-1])
            target_corpus.append([vec2word_target[w] for w in y[:-1]])
        return target_corpus

    def bptt(self, x, y):
        Tenc = len(x)
        Tdec = len(y)
        # Perform forward propagation
        h = self.encoder(x)
        c = h[-1]
        s,o = self.decoder(x, c, y)
        # We accumulate the gradients in these variables
        dLdUe = np.zeros(self.Ue.shape)
        dLdWe = np.zeros(self.We.shape)
        dLdP = np.zeros(self.P.shape)
        dLdV = np.zeros(self.V.shape)
        dLdUd = np.zeros(self.Ud.shape)
        dLdWd = np.zeros(self.Wd.shape)
        delta_c_t = activation_prime(c)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(Tdec)[::-1]:
            #print delta_o[t].shape, s[t].shape
            dLdP += np.outer(delta_o[t], s[t])
            # Initial delta calculation: dL/dz
            delta_s_t = np.dot(self.P.T, delta_o[t]) * activation_prime(s[t])
            dLdc = np.zeros(self.hidden_dim)
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for decoder_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                # Add to gradients at each previous step
                dLdUd += np.outer(delta_s_t, s[decoder_step - 1])
                dLdV += np.outer(delta_s_t, c)
                dLdWd[:, y[decoder_step - 1]] += delta_s_t
                dLdc += np.dot(self.V.T, delta_s_t)
                # Update delta for next step dL/dz in the decoder at t-1
                delta_s_t = np.dot(self.Ud.T, delta_s_t) * activation_prime(s[decoder_step - 1])
            delta_c_t = dLdc * activation_prime(c)
            for encoder_step in np.arange(max(0, Tenc - self.bptt_truncate), Tenc)[::-1]:
                dLdWe[:, x[encoder_step]] += delta_c_t
                dLdUe += np.outer(delta_c_t, h[encoder_step - 1])
                # Update delta for next step dL/dz in the decoder at t-1
                delta_c_t = np.dot(self.Ue.T, delta_c_t) * activation_prime(h[encoder_step - 1])

        return [dLdUe, dLdWe, dLdUd, dLdWd, dLdV, dLdP]


    def SGD(self,x,y,eta):
        dLdUe, dLdWe, dLdUd, dLdWd, dLdV, dLdP = self.bptt(x,y)
        self.Ue -= eta * dLdUe
        self.We -= eta * dLdWe
        self.Ud -= eta * dLdUd
        self.Wd -= eta * dLdWd
        self.V -= eta* dLdV
        self.P -= eta* dLdP

    def train_with_sgd(model, x, y, learning_rate=0.005, nepoch=10, evaluate_loss_after=1):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = model.total_loss(x,y)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print '%s: Loss after num_examples_seen=%d epoch=%d: %f' % (time, num_examples_seen, epoch, loss)
                # Adjust the learning rate if loss increases
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
                    print "Setting learning rate to %f" % learning_rate
                sys.stdout.flush()
            # For each training example...
            for i in range(len(y)):
                # One SGD step
                model.SGD(x[i], y[i], learning_rate)
                num_examples_seen += 1
        return losses

def activation(net):
    return 1.0/(1.0+np.exp(-net))

def activation_prime(z):
    return z*(1-z)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

train_X, word2vec_resource, vec2word_resource = getData('europarl-en-small')
train_Y, word2vec_target, vec2word_target = getData('europarl-de-small')
model1 = VanillaRnn(vocabulary_size,vocabulary_size,100)
model1.train_with_sgd(train_X[:50],train_Y[:50])
trial1 = model1.translate(train_X[50:52])
for i in range(len(trial1)):
    print " ".join([vec2word_resource[x] for x in train_X[50+i][:-1]])
    print " ".join(trial1[i])
    print " ".join([vec2word_target[x] for x in train_Y[50+i][:-1]])