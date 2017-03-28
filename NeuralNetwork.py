import random
import numpy as np

class Network(object):
    def __init__(self, sizes):
        """sizes = [2,3,1]  represents number of neurons in each layer"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.rand(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]


    def feed_forward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def cost_derivative_wrt_finalOutput(self, fo, y):
        return (fo - y)

    def backprop(self, x, y):

        """These will hold the nabla "δ" values wrt to individual biases and weights for each layer"""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        """Step 1 --  Calculate the feedforward `Net` and `Activation` values"""

        activation = x
        activations = [x]

        Nets = []

        for b, w in zip(self.biases, self.weights):
            net = np.dot(w, activation) + b
            Nets.append(net)
            activation = sigmoid(net)
            activations.append(activation)

        """Backward Pass

        Step 2  Calculate delta for the Output/Last Layer"""

        delta = derivative_sigmoid_wrt_Net(Nets[-1])
        delta = self.cost_derivative_wrt_finalOutput(activations[-1], y) * delta
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())


        """Step 3 Calculate deltas, nabal_b and nabla_w at each layer"""

        for layer in xrange(2, self.num_layers):
            net_layer = Nets[-layer]
            activation_layer =  derivative_sigmoid_wrt_Net(net_layer)
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * activation_layer
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer-1].transpose())


        return (nabla_b, nabla_w)


    def update_parameters_single_batch(self, batch, eta):
        """This method is like a single step of gradient descent"""

        """Initialize the matrices which will hold the sum of all the derivatives wrt parameters, for the whole batch"""

        derivatives_b_sum_batch = [np.zeros(b.shape) for b in self.biases]
        derivatives_w_sum_batch = [np.zeros(w.shape) for w in self.weights]

        """Pass each X and Y from the batch to backprop method to get their respective partial derivatives
        and then add them to the sum variables defined above one by one """

        for x, y in batch:
            delta_gradient_b, delta_gradient_w = self.backprop(x, y)
            derivatives_b_sum_batch = [db + dgb for db, dgb in zip(derivatives_b_sum_batch, delta_gradient_b)]
            derivatives_w_sum_batch = [dw + dwg for dw, dwg in zip(derivatives_w_sum_batch, delta_gradient_w)]

        """Now since we have the summation of the gradients of the entire batch, we need to update the weight and biases
        Matrcices by computing the formula:
        w -> w' = w - (eta/size_of_batch) * derivatives_sum_of_weights
        b -> b' = b - (eta/size_of_batch) * derivatives_sum_of_biases
        """

        size_of_batch = len(batch)
        self.biases = [b - ((eta/size_of_batch) * db)
                       for b, db in zip(self.biases, derivatives_b_sum_batch)]
        self.weights = [w - ((eta/size_of_batch) * dw)
                        for w, dw in zip(self.weights, derivatives_w_sum_batch)]


    def stochastic_Gradient_Descent(self, training_set, eta, epochs, batch_size, test_set = None):
        """This method takes the training set and splits the whole thing into a number of mini sets of each having
        size ```batch_size```  and then run the method update_parameters_single_batch() on each of these sets, instead
        of the whole training set at the same time. This operation is performed ```epochs``` number of times"""

        if test_set: test_set_size = len(test_set)
        training_set_size = len(training_set)

        #Generate epoch one by one
        for epochNo in xrange(epochs):
            random.shuffle(training_set)

            #Create batches each of size ```batch_size```
            batches = [training_set[batch_index:batch_index + batch_size]
                       for batch_index in xrange(0, training_set_size, batch_size)]

            #Update Parameters for each of these batches one by one
            for batch in batches:
                self.update_parameters_single_batch(batch, eta)

            """The following operations is just for logs and tracking progress and can make the program very slow
            if the test set is big enough"""

            if test_set:
                print "Epoch {0}: {1} / {2}".format(
                    epochNo, self.evaluate(test_set), test_set_size)
            else:
                print "Epoch {0} complete".format(epochNo)





    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)











def sigmoid(net):
    return 1.0 / (1.0 + np.exp(-net))

def derivative_sigmoid_wrt_Net(a):
    return sigmoid(a) * (1 - sigmoid(a))

