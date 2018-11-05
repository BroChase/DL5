# Chase Brown
# CSCI Deep Learning Program 5
# RNN/LSTM word based

import numpy as np
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import operator


def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

class RNNVanilla:

    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim  # size of the vocabulary
        self.hidden_dim = hidden_dim  # size of hidden layer
        self.bptt_truncate = bptt_truncate

        # Randomly initialize the network parameters of U V W
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)

        # During forward propagation we save all hidden states in s because need them later.

        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)

        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))

        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]  # We not only return the calculated outputs, but also the hidden states.
        # We will use them later to calculate the gradients

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0

        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])

            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]

            # Add to the loss based on how off we were
            L += -1 * sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N

    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)

            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))

            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t

                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return [dLdU, dLdV, dLdW]

    def gradient_check(self, x, y, f, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = self.bptt(x, y)

        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']

        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
            f.write("Performing gradient check for parameter %s with size %d.\n" % (pname, np.prod(parameter.shape)))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x], [y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x], [y])
                estimated_gradient = (gradplus - gradminus) / (2 * h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient) / (
                            np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                    print("+h Loss: %f" % gradplus)
                    print("-h Loss: %f" % gradminus)
                    print("Estimated_gradient: %f" % estimated_gradient)
                    print("Backpropagation gradient: %f" % backprop_gradient)
                    print("Relative Error: %f" % relative_error)
                    f.write("Gradient Check ERROR: parameter=%s ix=%s\n" % (pname, ix))
                    f.write("+h Loss: %f\n" % gradplus)
                    f.write("-h Loss: %f\n" % gradminus)
                    f.write("Estimated_gradient: %f\n" % estimated_gradient)
                    f.write("Backpropagation gradient: %f\n" % backprop_gradient)
                    f.write("Relative Error: %f\n" % relative_error)
                    return
                it.iternext()
            print("Gradient check for parameter %s passed." % (pname))
            f.write("Gradient check for parameter %s passed.\n" % (pname))

    # Performs one step of SGD.
    def sgd_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, index_to_char, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    strings = []
    num_examples_seen = 0

    for epoch in range(nepoch):
        if epoch == 20 or epoch == 40 or epoch == 60 or epoch == 80 or epoch == 100:
            predictions = model.predict(X_train[42])
            print(predictions.shape)
            print(predictions)
            strings.append(predictions)
            print("index_to_word>")
            print('%s' % " ".join([index_to_char[x] for x in predictions]))

        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
    return losses, strings


def lossvsepoch(epoch, loss, title):
    plt.plot(epoch, loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    t = title+'.png'
    plt.savefig(t)
    plt.show()


def runit(f, XTrain, yTrain, index_to_char, title, gphtitle, hiddendunits):
    # Train
    np.random.seed(10)
    # grad_check_vocab_size = 100
    # model = RNNVanilla(grad_check_vocab_size, hidden_dim=hiddendunits)
    model = RNNVanilla(len(index_to_char), hidden_dim=hiddendunits)
    # model.gradient_check([31, 40, 20, 43], [40, 20, 43, 50], f)

    print("Expected Loss for random predictions: %f" % np.log(len(index_to_char)))
    print("Actual loss: %f" % model.calculate_loss(XTrain, yTrain))
    f.write("Expected Loss for random predictions: %f\n" % np.log(len(index_to_char)))
    f.write("Actual loss: %f\n" % model.calculate_loss(XTrain, yTrain))
    losses, strings = train_with_sgd(model, XTrain, yTrain, index_to_char, learning_rate=0.005,
                                         nepoch=101, evaluate_loss_after=1)

    f.write(title)
    ep = [20, 40, 60, 80, 100]
    x_example, y_example = XTrain[42], yTrain[42]
    f.write('Example text\n')
    f.write("x:\n%s\n" % (" ".join([index_to_char[x] for x in x_example])))

    # Write out the strings that are returned by the model after training to see progress
    for i in range(len(strings)):
        print("index_to_word>")
        print('%s' % " ".join([index_to_char[x] for x in strings[i]]))
        f.write('Resulting outputs from training\n')
        f.write('epoch %d : ' % ep[i])
        f.write('%s\n' % " ".join([index_to_char[x] for x in strings[i]]))

    loss = []
    epoch = []
    for i in range(len(losses)):
        loss.append(losses[i][1])
        epoch.append(i)
    # Plot the epoch.loss
    lossvsepoch(epoch, loss, gphtitle)
    return loss, epoch