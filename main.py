# Chase Brown
# CSCI Deep Learning Program 5
# RNN/LSTM word based

import itertools
import numpy as np
import nltk
import rnn
import lstm
import os
import matplotlib.pyplot as plt

vocabulary_size = 5000
unknown_token = "UNKNOWN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# break up txt documents into sentences and concatenate them
# sentences = []
# for file in os.listdir('stories'):
#     with open('stories/'+file, encoding='utf8') as f:
#         lines = f.read()
#         s = nltk.sent_tokenize(lines)
#         for i in range(len(s)):
#             s[i] = s[i].replace('\n', ' ').lower()
#         sentences = sentences + s
#
# # Append SENTENCE_START and SENTENCE_END
# f = open('report.doc', 'w')
# sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
# print("Parsed %d sentences." % (len(sentences)))
# f.write('Parsed %d sentences.\n' % (len(sentences)))
#
# tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
#
# # Count the word frequencies
# word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
# print("Found %d unique words tokens." % len(word_freq.items()))
# f.write('Found %d unique word tokens.\n' % len(word_freq.items()))
#
# # Get the most common words and build index_to_word and word_to_index vectors
# vocab = word_freq.most_common(vocabulary_size-1)
# index_to_word = [x[0] for x in vocab]
# index_to_word.append(unknown_token)
# word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
#
# print("Using vocabulary size %d." % vocabulary_size)
# print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))
# print(word_freq.most_common(10))
# f.write("Using vocabulary size %d.\n" % vocabulary_size)
# f.write("The least frequent word in our vocabulary is '%s' and appeared %d times.\n" % (vocab[-1][0], vocab[-1][1]))
#
# # Replace all words not in our vocabulary with the unknown token
# for i, sent in enumerate(tokenized_sentences):
#     tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
#
# print("\nExample sentence: '%s'" % sentences[0])
# print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])
#
#
# # # Create the training data
# # XTrain = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
# # yTrain = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
#
# # ++++++++++++++++++++++++Gradient Check++++++++++++++++++++++++++++++++++++++++++++
# # np.random.seed(10)
# # grad_check_vocab_size = 100
# # model = rnn.RNNVanilla(grad_check_vocab_size, 10, bptt_truncate=1000)  #todo print out gradient check
# # model.gradient_check([31, 40, 20, 43], [40, 20, 43, 50], f)
#
# # model = rnn.RNNVanilla(len(index_to_word))
# # # Limit to 1000 examples to save time
# # print("Expected Loss for random predictions: %f" % np.log(len(index_to_word)))
# # print("Actual loss: %f" % model.calculate_loss(XTrain[:1000], yTrain[:1000]))
# # f.write("Expected Loss for random predictions: %f\n" % np.log(len(index_to_word)))
# # f.write("Actual loss: %f\n" % model.calculate_loss(XTrain[:1000], yTrain[:1000]))
# # ========================rnn vanilla 100 hidden units==============================
# # Create the training data length of 25
# toke_n = []
# for i in range(len(tokenized_sentences)):
#     if len(tokenized_sentences[i]) == 10:
#         toke_n.append(tokenized_sentences[i])
#
# XTrain = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in toke_n])
# yTrain = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in toke_n])
# loss = []
# # Train
# f.write('rnnVanilla with 100 Hidden Units & sequence length of 25:\n') #todo edit length
# L = rnn.runit(f, XTrain, yTrain, index_to_word, 'rnnVanilla 100 Hidden Units\n', '100hidden', 100)
# loss.append(L)
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # ========================rnn vanilla 50 hidden units==================================================================
# f.write('rnnVanilla with 50 Hidden Units & sequence length of 25:\n')
# L = rnn.runit(f, XTrain, yTrain, index_to_word, 'rnnVanilla 50 Hidden Units\n', '50hidden', 50)
# loss.append(L)
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # ========================rnn vanilla 200 hidden units=================================================================
# f.write('rnnVanilla with 200 Hidden Units & sequence length of 25:\n')
# L = rnn.runit(f, XTrain, yTrain, index_to_word, 'rnnVanilla 200 Hidden Units\n', '200hidden', 200)
# loss.append(L)
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# toke_n = []
# for i in range(len(tokenized_sentences)):
#     if len(tokenized_sentences[i]) == 5:
#         toke_n.append(tokenized_sentences[i])
#
# # Create the training data for the new sequence
# XTrain = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in toke_n])
# yTrain = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in toke_n])
# # ========================rnn vanilla 100 hidden layers/half sequence=================
# f.write('rnnVanilla with 100 Hidden Units & sequence length of 12:\n')
# L = rnn.runit(f, XTrain, yTrain, index_to_word, 'rnnVanilla Half Sequence\n', 'HalfSeq', 100)
# loss.append(L)
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# toke_n = []
# for i in range(len(tokenized_sentences)):
#     if len(tokenized_sentences[i]) == 20:
#         toke_n.append(tokenized_sentences[i])
#
# # Create the training data for the new sequence
# XTrain = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in toke_n])
# yTrain = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in toke_n])
# # ========================rnn vanilla 100 hidden layers/half sequence=================
# f.write('rnnVanilla with 100 Hidden Units & sequence length of 50:\n')
# L, epoch = rnn.runit(f, XTrain, yTrain, index_to_word, 'rnnVanilla Double Sequence\n', 'DoubleSeq', 100)
# loss.append(L)
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# f.close()
#
# # ********************************LSTM START****************************************************************************
# break up txt documents into sentences and concatenate them
sentences = []
for file in os.listdir('stories'):
    with open('stories/'+file, encoding='utf8') as f:
        lines = f.read()
        s = nltk.sent_tokenize(lines)
        for i in range(len(s)):
            s[i] = s[i].replace('\n', ' ').lower()
        sentences = sentences + s

# Append SENTENCE_START and SENTENCE_END
f = open('reportlstm.doc', 'w')
sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))
f.write('Found %d unique word tokens.\n' % len(word_freq.items()))

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
idx_to_word = [x[0] for x in vocab]
idx_to_word.append(unknown_token)
word_to_idx = dict([(w, i) for i, w in enumerate(idx_to_word)])

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_idx else unknown_token for w in sent]

# Create the training data length of 25
toke_n = []
for i in range(len(tokenized_sentences)):
    if len(tokenized_sentences[i]) == 25:
        toke_n.append(tokenized_sentences[i])


X_size = vocabulary_size
data_size = sentences

print(data_size)
print(X_size)
print(word_to_idx)
print(idx_to_word)
# Constants and hyper-params
H_size = 100  # Size of the hidden layer
T_steps = 25  # Number of time steps (length of the sequence) used for training
learning_rate = 1e-1  # Learning rate
weight_sd = 0.1  # Standard deviation of weights for initialization
z_size = H_size + X_size  # Size of concatenate(H, X) vector
# inputs = np.asarray([[word_to_idx[w] for w in sent[:-1]] for sent in tokenized_sentences])
# targets = np.asarray([[word_to_idx[w] for w in sent[1:]] for sent in tokenized_sentences])
lstm.lstm(f, H_size, z_size, weight_sd, X_size, learning_rate, T_steps, toke_n, idx_to_word, word_to_idx, 'lstm100hidden')
H_size = 50
z_size = H_size + X_size
lstm.lstm(f, H_size, z_size, weight_sd, X_size, learning_rate, T_steps, toke_n, idx_to_word, word_to_idx, 'lstm50hidden')
H_size = 200
z_size = H_size + X_size
lstm.lstm(f, H_size, z_size, weight_sd, X_size, learning_rate, T_steps, toke_n, idx_to_word, word_to_idx, 'lstm200hidden')
H_size = 100
z_size = H_size + X_size
T_steps = 12
# Create the training data length of 12
toke_n = []
for i in range(len(tokenized_sentences)):
    if len(tokenized_sentences[i]) == 12:
        toke_n.append(tokenized_sentences[i])
lstm.lstm(f, H_size, z_size, weight_sd, X_size, learning_rate, T_steps, toke_n, idx_to_word, word_to_idx, 'lstmSeq15')
T_steps = 50
# Create the training data length of 50
toke_n = []
for i in range(len(tokenized_sentences)):
    if len(tokenized_sentences[i]) == 50:
        toke_n.append(tokenized_sentences[i])
lstm.lstm(f, H_size, z_size, weight_sd, X_size, learning_rate, T_steps, toke_n, idx_to_word, word_to_idx, 'lstmSeq50')

f.close()

