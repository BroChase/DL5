# Chase Brown
# CSCI Deep Learning Program 4
# RNN/LSTM word based

import numpy as np
import signal
import rnn
from random import uniform

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1 - y)


def tanh(x):
    return np.tanh(x)


def dtanh(y):
    return 1 - y * y


# Parameters
# We use random weights with normal distribution (0, weight_sd) for  tanh  activation function
# and (0.5, weight_sd) for  sigmoid  activation function.
# Biases are initialized to zeros.
class Param:
    def __init__(self, name, value):
        self.name = name
        self.v = value  # parameter value
        self.d = np.zeros_like(value)  # derivative
        self.m = np.zeros_like(value)  # momentum for AdaGrad


class Parameters:
    def __init__(self, H_size, z_size, weight_sd, X_size):
        self.W_f = Param('W_f', np.random.randn(H_size, z_size) * weight_sd + 0.5)
        self.b_f = Param('b_f', np.zeros((H_size, 1)))

        self.W_i = Param('W_i', np.random.randn(H_size, z_size) * weight_sd + 0.5)
        self.b_i = Param('b_i', np.zeros((H_size, 1)))

        self.W_C = Param('W_C', np.random.randn(H_size, z_size) * weight_sd)
        self.b_C = Param('b_C', np.zeros((H_size, 1)))

        self.W_o = Param('W_o', np.random.randn(H_size, z_size) * weight_sd + 0.5)
        self.b_o = Param('b_o', np.zeros((H_size, 1)))

        # For final layer to predict the next character
        self.W_v = Param('W_v', np.random.randn(X_size, H_size) * weight_sd)
        self.b_v = Param('b_v', np.zeros((X_size, 1)))

    def all(self):
        return [self.W_f, self.W_i, self.W_C, self.W_o, self.W_v,
                self.b_f, self.b_i, self.b_C, self.b_o, self.b_v]

class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        print('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


class lstm:
    def __init__(self, f, H_size, z_size, weight_sd, X_size, learning_rate, T_steps, data, idx_to_char, char_to_idx, title):
        global plot_iter, plot_loss
        global smooth_loss
        global parameters
        H_size = H_size
        z_size = z_size
        weight_sd = weight_sd
        X_size = X_size
        learning_rate = learning_rate
        T_steps = T_steps
        data = data
        idx_to_char = idx_to_char
        char_to_idx = char_to_idx
        title = title

        parameters = Parameters(H_size=H_size, z_size=z_size, weight_sd=weight_sd, X_size=X_size)


        def forward(x, h_prev, C_prev, p=parameters):
            assert x.shape == (X_size, 1)
            assert h_prev.shape == (H_size, 1)
            assert C_prev.shape == (H_size, 1)

            z = np.row_stack((h_prev, x))
            f = sigmoid(np.dot(p.W_f.v, z) + p.b_f.v)
            i = sigmoid(np.dot(p.W_i.v, z) + p.b_i.v)
            C_bar = tanh(np.dot(p.W_C.v, z) + p.b_C.v)

            C = f * C_prev + i * C_bar
            o = sigmoid(np.dot(p.W_o.v, z) + p.b_o.v)
            h = o * tanh(C)

            v = np.dot(p.W_v.v, h) + p.b_v.v
            y = np.exp(v) / np.sum(np.exp(v))  # softmax

            return z, f, i, C_bar, C, o, h, v, y


        def backward(target, dh_next, dC_next, C_prev, z, f, i, C_bar, C, o, h, v, y, p=parameters):
            assert z.shape == (X_size + H_size, 1)
            assert v.shape == (X_size, 1)
            assert y.shape == (X_size, 1)

            for param in [dh_next, dC_next, C_prev, f, i, C_bar, C, o, h]:
                assert param.shape == (H_size, 1)

            dv = np.copy(y)
            dv[target] -= 1

            p.W_v.d += np.dot(dv, h.T)
            p.b_v.d += dv

            dh = np.dot(p.W_v.v.T, dv)
            dh += dh_next
            do = dh * tanh(C)
            do = dsigmoid(o) * do
            p.W_o.d += np.dot(do, z.T)
            p.b_o.d += do

            dC = np.copy(dC_next)
            dC += dh * o * dtanh(tanh(C))
            dC_bar = dC * i
            dC_bar = dtanh(C_bar) * dC_bar
            p.W_C.d += np.dot(dC_bar, z.T)
            p.b_C.d += dC_bar

            di = dC * C_bar
            di = dsigmoid(i) * di
            p.W_i.d += np.dot(di, z.T)
            p.b_i.d += di

            df = dC * C_prev
            df = dsigmoid(f) * df
            p.W_f.d += np.dot(df, z.T)
            p.b_f.d += df

            dz = (np.dot(p.W_f.v.T, df) + np.dot(p.W_i.v.T, di) + np.dot(p.W_C.v.T, dC_bar) + np.dot(p.W_o.v.T, do))
            dh_prev = dz[:H_size, :]
            dC_prev = f * dC

            return dh_prev, dC_prev


        # Clear gradients before each backward pass
        def clear_gradients(params=parameters):
            for p in params.all():
                p.d.fill(0)


        # Clip gradients to mitigate exploding gradients
        def clip_gradients(params=parameters):
            for p in params.all():
                np.clip(p.d, -1, 1, out=p.d)


        def forward_backward(inputs, targets, h_prev, C_prev):
            global paramters

            # To store the values for each time step
            x_s, z_s, f_s, i_s, = {}, {}, {}, {}
            C_bar_s, C_s, o_s, h_s = {}, {}, {}, {}
            v_s, y_s = {}, {}

            # Values at t - 1
            h_s[-1] = np.copy(h_prev)
            C_s[-1] = np.copy(C_prev)

            loss = 0
            # Loop through time steps
            assert len(inputs) == T_steps
            for t in range(len(inputs)):
                x_s[t] = np.zeros((X_size, 1))
                x_s[t][inputs[t]] = 1  # Input character

                (z_s[t], f_s[t], i_s[t], C_bar_s[t], C_s[t], o_s[t], h_s[t], v_s[t], y_s[t]) = \
                    forward(x_s[t], h_s[t - 1], C_s[t - 1])  # Forward pass

                loss += -np.log(y_s[t][targets[t], 0])  # Loss for at t

            clear_gradients()

            dh_next = np.zeros_like(h_s[0])  # dh from the next character
            dC_next = np.zeros_like(C_s[0])  # dh from the next character

            for t in reversed(range(len(inputs))):
                # Backward pass
                dh_next, dC_next = backward(target=targets[t], dh_next=dh_next, dC_next=dC_next, C_prev=C_s[t - 1],
                             z=z_s[t], f=f_s[t], i=i_s[t], C_bar=C_bar_s[t], C=C_s[t], o=o_s[t], h=h_s[t], v=v_s[t], y=y_s[t])

            clip_gradients()

            return loss, h_s[len(inputs) - 1], C_s[len(inputs) - 1]


        def generate_next_char(h_prev, C_prev, first_char_idx, sentence_length):
            x = np.zeros((X_size, 1))
            x[first_char_idx] = 1

            h = h_prev
            C = C_prev

            indexes = []

            for t in range(sentence_length):
                _, _, _, _, C, _, h, _, p = forward(x, h, C)
                idx = np.random.choice(range(X_size), p=p.ravel())
                x = np.zeros((X_size, 1))
                x[idx] = 1
                indexes.append(idx)

            return indexes


    # Update the graph and display a sample output
        def update_status(inputs, h_prev, C_prev):
            # initialized later

            # Get predictions for 200 words with current model

            sample_idx = generate_next_char(h_prev, C_prev, inputs[0], 25)
            txt = ' '.join(idx_to_char[idx] for idx in sample_idx)

            # Clear and plot
            # plt.plot(plot_iter, plot_loss)
            # display.clear_output(wait=True)
            # plt.show()

            # Print prediction and loss
            print("----\n %s \n----" % txt)
            print("iter %d, loss %f" % (iteration, smooth_loss))

            f.write("iter %d, loss %f\n" % (iteration, smooth_loss))
            f.write("----\n %s \n----\n\n" % txt)

        def update_paramters(params = parameters):
            for p in params.all():
                p.m += p.d * p.d # Calculate sum of gradients
                #print(learning_rate * dparam)
                p.v += -(learning_rate * p.d / np.sqrt(p.m + 1e-8))

        # To delay the keyboard interrupt to prevent the training from stopping in the middle of an iteration
        # Exponential average of loss
        # Initialize to a error of a random model
        smooth_loss = -np.log(1.0 / X_size) * T_steps

        iteration, pointer = 0, 0

        # For the graph
        plot_iter = np.zeros(0)
        plot_loss = np.zeros(0)

        # while True:
        #     try:
        #         with DelayedKeyboardInterrupt():
        #             # Reset
        #             if pointer + T_steps >= len(data) or iteration == 0:
        #                 g_h_prev = np.zeros((H_size, 1))
        #                 g_C_prev = np.zeros((H_size, 1))
        #                 pointer = 0
        #
        #             inputs = ([char_to_idx[ch]for ch in data[pointer: pointer + T_steps]])
        #             targets = ([char_to_idx[ch]for ch in data[pointer + 1: pointer + T_steps + 1]])
        #
        #             loss, g_h_prev, g_C_prev = forward_backward(inputs, targets, g_h_prev, g_C_prev)
        #             smooth_loss = smooth_loss * 0.999 + loss * 0.001
        #
        #             # Print every hundred steps
        #             if iteration % 100 == 0:
        #                 update_status(inputs, g_h_prev, g_C_prev)
        #
        #             update_paramters()
        #
        #             plot_iter = np.append(plot_iter, [iteration])
        #             plot_loss = np.append(plot_loss, [loss])
        #
        #             pointer += T_steps
        #             iteration += 1
        #     except KeyboardInterrupt:
        #         update_status(inputs, g_h_prev, g_C_prev)
        #         break
        while iteration != 101:
            # if pointer + T_steps >= len(data) or iteration == 0:
            #     g_h_prev = np.zeros((H_size, 1))
            #     g_C_prev = np.zeros((H_size, 1))
            #     pointer = 0

            if pointer >= len(data) or iteration == 0:
                g_h_prev = np.zeros((H_size, 1))
                g_C_prev = np.zeros((H_size, 1))
                pointer = 0

            #inputs = ([char_to_idx[word] for word in data[pointer: pointer + T_steps]])
            inputs = ([char_to_idx[w] for w in data[pointer]])
            #targets = ([char_to_idx[ch] for ch in data[pointer + 1: pointer + T_steps + 1]])
            targets = ([char_to_idx[w] for w in data[pointer]])

            loss, g_h_prev, g_C_prev = forward_backward(inputs, targets, g_h_prev, g_C_prev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            # Print every hundred steps
            # if iteration % 100 == 0:
            #     update_status(inputs, g_h_prev, g_C_prev)
            if iteration == 20 or iteration == 40 or iteration == 60 or iteration == 80 or iteration == 100:
                update_status(inputs, g_h_prev, g_C_prev)

            update_paramters()

            plot_iter = np.append(plot_iter, [iteration])
            plot_loss = np.append(plot_loss, [loss])

            pointer += 1
            iteration += 1


        # Calculate numerical gradient
        def calc_numerical_gradient(param, idx, delta, inputs, target, h_prev, C_prev):
            old_val = param.v.flat[idx]

            # evaluate loss at [x + delta] and [x - delta]
            param.v.flat[idx] = old_val + delta
            loss_plus_delta, _, _ = forward_backward(inputs, targets, h_prev, C_prev)
            param.v.flat[idx] = old_val - delta
            loss_mins_delta, _, _ = forward_backward(inputs, targets, h_prev, C_prev)

            param.v.flat[idx] = old_val  # reset

            grad_numerical = (loss_plus_delta - loss_mins_delta) / (2 * delta)
            # Clip numerical error because analytical gradient is clipped
            [grad_numerical] = np.clip([grad_numerical], -1, 1)

            return grad_numerical


        # Check gradient of each paramter matrix/vector at `num_checks` individual values
        def gradient_check(num_checks, delta, inputs, target, h_prev, C_prev):

            # To calculate computed gradients
            _, _, _ = forward_backward(inputs, targets, h_prev, C_prev)

            for param in parameters.all():
                # Make a copy because this will get modified
                d_copy = np.copy(param.d)

                # Test num_checks times
                for i in range(num_checks):
                    # Pick a random index
                    rnd_idx = int(uniform(0, param.v.size))

                    grad_numerical = calc_numerical_gradient(param, rnd_idx, delta, inputs, target, h_prev, C_prev)
                    grad_analytical = d_copy.flat[rnd_idx]

                    err_sum = abs(grad_numerical + grad_analytical) + 1e-09
                    rel_error = abs(grad_analytical - grad_numerical) / err_sum

                    # If relative error is greater than 1e-06
                    if rel_error > 1e-06:
                        print('%s (%e, %e) => %e' % (param.name, grad_numerical, grad_analytical, rel_error))
                        f.write('%s (%e, %e) => %e' % (param.name, grad_numerical, grad_analytical, rel_error))


        print(gradient_check(10, 1e-5, inputs, targets, g_h_prev, g_C_prev))
        # f.write(gradient_check(10, 1e-5, inputs, targets, g_h_prev, g_C_prev))

        # Clear and plot
        rnn.lossvsepoch(plot_iter, plot_loss, title)
        # Print prediction and loss
        print("iter %d, loss %f" % (iteration, smooth_loss))
        f.write("iter %d, loss %f\n\n" % (iteration, smooth_loss))