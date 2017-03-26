
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

    def encoder(self, x):
    # The total number of time steps
    Tenc = len(x)
    # During forward propagation we save all hidden states in h because need them for back propagation.
    h = np.zeros((Tenc + 1, self.hidden_dim))
    # For each time step...
    for t in np.arange(Tenc):
        # Note that we are indexing W by x[t]. This is the same as multiplying U with a one-hot vector.
        h[t] = activation(self.We[:,x[t]] + np.dot(self.Ue,h[t-1]))
    return h

    def decoder(self, x, c, y=None):
    #Training case
    if (y!=None):
        # The total number of time steps
        Tdec = len(y)
        # During forward propagation we save all hidden states in h because need them for back propagation.
        s = np.zeros((Tdec + 1, self.hidden_dim))
        o = np.zeros((Tdec, self.hidden_dim))
        # For each time step...
        for t in np.arange(Tdec):
        # Note that we are indexing W by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = activation(self.W[:,y[t-1,]] + np.dot(self.U,s[t-1])+ np.dot(self.V,c))
            o[t] = softmax(np.dot(self.P, s[t]))
        return [s,o]
    #Translation case
    else:
        #Start translation without previous word presented
        s = [activation(np.dot(self.V,c))]
        o = [softmax(np.dot(self.P, s))]
        y = [np.argmax(o)]
        #Keep translation till the ending sign is reached
        while(y[-1]!=sentence_end_token):
            s.append(activation(self.W[:,y[t-1,]] + np.dot(self.U,s[t-1])+ np.dot(self.V,c)))
            o.append(softmax(np.dot(self.P, s[t])))
            y.append(np.argmax(o[t]))
            t = t+1
        y = np.asarray(y)
        o = np.asarray(o)
        s = np.asarray(s)
    return [s,o,y]

    def bptt(self, x, y):
    Tenc = len(x)
    Tdec = len(y)
    # Perform forward propagation
    h = self.endcoder(x)
    c = h[-1]
    o, s = self.decoder(x,c,y)
    # We accumulate the gradients in these variables
    dLdUe = np.zeros(self.Ue.shape)
    dLdWe = np.zeros(self.We.shape)
    dLdP = np.zeros(self.V.shape)
    dLdV = np.zeros(self.P.shape)
    dLdUd = np.zeros(self.Ud.shape)
    dLdWd = np.zeros(self.Wd.shape)
    delta_c_t = activation_prime(c)
    for step in np.arange(max(0, T-self.bptt_truncate), T)[::-1]:
        dcdWe[:,x[step]] += delta_c_t
        dcdUe += np.outer(delta_c_t,h[step-1])
        # Update delta for next step dL/dz at t-1
        delta_c_t +=np.dot(self)
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1.
    # For each output backwards...
    for t in np.arange(Tdec)[::-1]:
        dLdP += np.outer(delta_o[t], s[t].T)
        # Initial delta calculation: dL/dz
        delta_s_t = np.dot(self.P.T,delta_o[t]) * activation_prime(s[t])
        # Backpropagation through time (for at most self.bptt_truncate steps)
        for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
            # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            # Add to gradients at each previous step 
            dLdUd += np.outer(delta_s_t, s[bptt_step-1])
            dLdV += np.outer(delta_s_t, c)
            dLdWd[:,y[bptt_step-1]] += delta_s_t
            dLdc += np.dot(self.V.T,delta_s_t)
            # Update delta for next step dL/dz in the decoder at t-1
            delta_s_t = np.dot(self.Ud.T, delta_s_t) * activation_prime(s[bptt_step-1])
        delta_c_t = dLdc*activation_prime(c)
        for bptt_step in np.arange(max(0, Tenc-self.bptt_truncate), Tenc)[::-1]:
            dLdWe[:,x[step]] += delta_c_t
            dLdUe += np.outer(delta_c_t,h[step-1])
            # Update delta for next step dL/dz in the decoder at t-1
            delta_c_t = np.dot(self.Ue.T, delta_c_t) * activation_prime(h[bptt_step-1]) 
        
            
    return [dLdUe, dLdWe, dLdUd, dLdWd, dLdV, dLdP]



