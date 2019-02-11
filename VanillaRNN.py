import numpy as np
from test import *
from datetime import datetime
import sys
import os
import time
#import math

class VanillaRnn(object):
    
    def __init__(self, source_dim, target_dim, hidden_dim, bptt_truncate=5):
        # Assign dimensions to input, output and hidden layers
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
        # Divide the total loss by the number of tokens in training sets
        L = 0
        N = 0
        for i in np.arange(len(y)):
            L += self.calculate_loss(x[i],y[i])
            N += len(y[i])
        return L / N

    def encoder(self, x):
        # The total number of time steps in our encoder
        Tenc = len(x)
        # During forward propagation we keep track to all hidden states h for back propagation.
        h = np.zeros((Tenc, self.hidden_dim))
        # For each time step...
        for t in np.arange(Tenc):
            # Note that we are indexing W by x[t]. This is the same as multiplying W with a one-hot vector.
            h[t] = activation(self.We[:, x[t]] + np.dot(self.Ue, h[t-1]))
        return h

    def decoder(self, x, c, y=None):
        if (y != None):# Training case
            # The total number of time steps
            Tdec = len(y)
            # During forward propagation we keep track to all hidden states s for back propagation.
            s = np.zeros((Tdec, self.hidden_dim))
            s[0] = activation(np.dot(self.V, c))
            o = np.zeros((Tdec, self.target_dim))
            o[0] = softmax(np.dot(self.P, s[0]))
            # For each time step...
            for t in np.arange(1,Tdec):
                # Note that we are indexing W by y[t-1]. This is the same as multiplying W with a one-hot vector.
                s[t] = activation(self.Wd[:, y[t - 1]] + np.dot(self.Ud, s[t - 1]) + np.dot(self.V, c))
                o[t] = softmax(np.dot(self.P, s[t]))
            return [s, o]
        else:# Translation case
            # Start translation without previous word presented
            s = [activation(np.dot(self.V, c))]
            o = [softmax(np.dot(self.P, s[0]))]
            y = [np.argmax(o)]
            t = 1
            # Keep translation till the ending sign is reached
            while y[-1] != word2vec_target[sentence_end_token] and len(y)<30 :
                s.append(activation(self.Wd[:, y[t - 1]] + np.dot(self.Ud, s[t - 1]) + np.dot(self.V, c)))
                o.append(softmax(np.dot(self.P, s[t])))
                #trial = word2vec_target[unknown_token]
                trial = word2vec_target[np.argmax(o[t])]
                # Resample until no "UNKNOWN_TOKEN" is drawn
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
        #tf_itf = tf[y] * itf[y]
        #delta_o = [(1+tf_itf)*delta_o[t] for t in np.arange(len(y))]
        delta_o = np.asarray([weight*delta_o[t] for t in np.arange(len(y))])
        # For each output backwards...
        for t in np.arange(Tdec)[::-1]:
            #print delta_o[t].shape, s[t].shape
            dLdP += np.outer(delta_o[t], s[t])
            # Initial delta calculation: dL/dz
            delta_s_t = np.dot(self.P.T, delta_o[t]) * activation_prime(s[t])
            dLdc = np.zeros(self.hidden_dim)
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for decoder_step in np.arange(max(1, t - self.bptt_truncate), t + 1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                # Add to gradients at each previous step
                dLdUd += np.outer(delta_s_t, s[decoder_step - 1])
                dLdV += np.outer(delta_s_t, c)
                dLdWd[:, y[decoder_step - 1]] += delta_s_t
                dLdc += np.dot(self.V.T, delta_s_t)
                # Update delta for next step dL/dz in the decoder at t-1
                delta_s_t = np.dot(self.Ud.T, delta_s_t) * activation_prime(s[decoder_step - 1])
            if t-self.bptt_truncate < 1 :
                dLdV += np.outer(delta_s_t, c)
                dLdc += np.dot(self.V.T, delta_s_t)
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
        examples_used = 0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if epoch % evaluate_loss_after == 0:
                loss = model.total_loss(x,y)
                losses.append((examples_used, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print '%s: Loss after %d examples used in model and epoch=%d: %f' % (time, examples_used, epoch, loss)
                # Decrease the learning rate if the current loss increases
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learning_rate = learning_rate * 0.5
                    print "Setting learning rate to %f" % learning_rate
                sys.stdout.flush()
            # For each training example...
            for i in range(len(y)):
                # One SGD step
                model.SGD(x[i], y[i], learning_rate)
                examples_used += 1
        return losses

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients by our bptt algorithm.
        # We want to check if them are correct.
        bptt_gradients = self.bptt(x, y)
        # List of all parameters pending to be checked.
        model_parameters = ['Ue', 'We', 'Ud', 'Wd', 'V', 'P']
        # Gradient check for each parameter
        for pari, parn in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.Ue
            pars = operator.attrgetter(parn)(self)
            total = 0
            failed = 0
            print "Performing gradient check for parameter %s with size %d." % (parn, np.prod(pars.shape))
            # Iterate over each element (i,j) of the parameter matrix,from (0,0) to (n-1,n-1).
            it = np.nditer(pars, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                total +=1
                ix = it.multi_index
                # Save the original value so we can reset it later
                center_point = pars[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                pars[ix] = center_point + h
                fplus = self.calculate_loss(x, y)
                pars[ix] = center_point - h
                fminus = self.calculate_loss(x, y)
                delta_gradient = (fplus - fminus) / (2 * h)
                # Reset parameter to original value
                pars[ix] = center_point
                # The gradient for this parameter calculated using backpropagation
                target_gradient = bptt_gradients[pari][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                if target_gradient==0.0 and delta_gradient==0.0:
                    relative_error=0
                else:
                    relative_error = np.abs(1000*target_gradient - 1000*delta_gradient) / \
                                 (1000*np.abs(target_gradient) + 1000*np.abs(delta_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    failed +=1
                    print "Gradient Check ERROR: parameter=%s ix=%s" % (parn, ix)
                    print "Estimated_gradient: %f" % delta_gradient
                    print "Backpropagation gradient: %f" % target_gradient
                    print "Relative Error: %f" % relative_error
                    #return
                it.iternext()
            print "%d of %d elements passed gradient check for parameter %s." % (total-failed,total,parn)






def activation(net):
    return 1.0/(1.0+np.exp(-net))

def activation_prime(z):
    return z*(1-z)
"""
def activation(net):
    return (np.exp(net)-1.0)/(1.0+np.exp(net))

def activation_prime(z):
    return 1-z**2
"""
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



train_XX, train_YY = load_dataset('europarl-en-small','europarl-de-small')
train_X, word2vec_resource, vec2word_resource, vocab_source, sentence_freq_source= train_XX[0], \
                                                                                   train_XX[1], train_XX[2], train_XX[3], train_XX[4]
train_Y, word2vec_target, vec2word_target, vocab_target, sentence_freq_target = train_YY[0], \
                                                                                   train_YY[1], train_YY[2], train_YY[3], train_YY[4]

words_freq_target = [x[1] for x in vocab_target]
words_freq_target.append(0)
words_freq_target=np.asarray(words_freq_target)

sentence_freq_target = [x[1] for x in sentence_freq_target]
sentence_freq_target.append(1)
sentence_freq_target=np.asarray(sentence_freq_target)

tf = words_freq_target/float(np.sum(words_freq_target))
itf = np.log(doc_size/(sentence_freq_target.astype(float)+1))
tf_itf =tf*itf
# Print tf_itf value for the top 20 frequent tokens
#print tf_itf[:20]
weight = itf+1

np.random.seed(10)
model1 = VanillaRnn(vocabulary_size,vocabulary_size,50)
"""
np.random.seed(10)
model_test = VanillaRnn(5,5,2)

model_test.gradient_check([0,1,3,2],[0,3,4,2])

"""
model1.train_with_sgd(train_X[:3000],train_Y[:3000])
trial1 = model1.translate(train_X[[2000,4000]])
for i in range(len(trial1)):
    print " ".join([vec2word_resource[x] for x in train_X[2000+i][:-1]])
    print " ".join(trial1[i])
    print " ".join([vec2word_target[x] for x in train_Y[2000+i][:-1]])


