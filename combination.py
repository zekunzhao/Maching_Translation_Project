from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import os
import random
import operator
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from queue import PriorityQueue
import sacrebleu
# from bfbs.datastructures.min_max_queue import MinMaxHeap
from nltk.translate.bleu_score import sentence_bleu


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass


def long_function_call():
    a = 0
    while 1:
        a = a+1
        # print("running!")
    return

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)



SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, currentList, wordId, score, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.score = score
        self.leng = length
        self.currentList = currentList
    def __lt__(self, other):
        return self.score < other.score
    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.score

        # return -self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
class DFS_exp_score(object):
    def __init__(self,thresh_score,encoder_outputs,decoder):

        self.best_node = None
        self.best_score = thresh_score
        self.encoder_outputs = encoder_outputs
        self.decoder = decoder

    def dfs(self,root):
        self._dfs(root)
        return self.best_score, self.best_node

    def _dfs(self,n):
        # print("deeper::",output_lang.index2word[n.wordid.item()])
        # print("deeper::",output_lang.index2word[n.wordid.item()])
        # print(n.leng)
        # wait = input("PRESS ENTER TO CONTINUE.")

        if n.wordid.item() == EOS_token or n.leng == MAX_LENGTH:
            return n.score, n
        decoder_input = n.wordid
        decoder_hidden = n.h
        decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, self.encoder_outputs)
        # for index in range(10):
        #     print(decoder_output[0][index])
        topv, topi = (-decoder_output).topk(len(decoder_output[0]))
        # for new_k in range(len(decoder_output[0])):
        #     if topi[0][new_k].squeeze().item() < 10:
        #         print("testing!!-----------------")
        #         print(topi[0][new_k].squeeze().item())
        #         print(-topv[0][new_k])
        # wait = input("PRESS ENTER TO CONTINUE.")

    
        # if self.best_score <= n.score-torch.exp(decoder_output[0][EOS_token]):
        #     print()
        #     print("Original thresh_score ====>  ",self.best_score)
            
        #     currentList = n.currentList.copy()
        #     currentList.append(EOS_token)
        #     self.best_score = n.score-torch.exp(decoder_output[0][EOS_token])
        #     endnode = BeamSearchNode(decoder_hidden, n, currentList, torch.tensor(EOS_token), self.best_score, n.leng + 1)
        #     self.best_node = endnode
        #     print("Updating thresh_score ====>  ",self.best_score)
        #     print(self.best_node.currentList)

        EOS_token_score = n.score-torch.exp(decoder_output[0][EOS_token])
        if EOS_token_score >=  self.best_score:
            # print()
            # print("Original thresh_score ====>  ",self.best_score)
            # self.second_node = self.best_node
            
            currentList = n.currentList.copy()
            currentList.append(EOS_token)
            self.best_score = EOS_token_score
            endnode = BeamSearchNode(decoder_hidden, n, currentList, torch.tensor(EOS_token), self.best_score, n.leng + 1)
            self.best_node = endnode
            print("serching ====>", '{:<30}'.format(str(self.best_node.currentList)) , "  score : ", "{:.2f}".format(self.best_score.item()))
    
    
    
        for new_k in range(len(decoder_output[0])):

    
            decoded_t = topi[0][new_k].squeeze().detach()
            score = -torch.exp(-topv[0][new_k])
            currentList = n.currentList.copy()
            currentList.append(decoded_t.item())
            new_score = n.score + score
            if new_score <  self.best_score:
                return self.best_score, self.best_node



            node = BeamSearchNode(decoder_hidden, n, currentList, decoded_t, new_score, n.leng + 1)
            # print("\rserching ====>", '{:<30}'.format(str(node.currentList)) ,"    score : ", "{:.2f}".format(node.score.item()) , "  thresh : ", "{:.2f}".format(self.best_score.item()),end="",flush=True)
            endscore, endnode = self._dfs(node)

    
    
            # print("new_score",new_score)
            # print("thresh_score",thresh_score)
            # wait = input("PRESS ENTER TO CONTINUE.")
            if endscore > self.best_score:
                
                # print("endnode--->", output_lang.index2word[endnode.wordid.item()])
                # print("endscore--->", endscore)
                # print()
                # print("Original thresh_score ====>  ",self.best_score)
                self.best_score = endscore
                self.best_node = endnode
                # print("Updating thresh_score ====>  ",self.best_score)
                # if self.best_node != None:
                #     print(self.best_node.currentList)

        return self.best_score, self.best_node
class DFS(object):
    def __init__(self,thresh_score,encoder_outputs,decoder):

        self.best_node = None
        self.best_score = thresh_score
        self.encoder_outputs = encoder_outputs
        self.decoder = decoder

    def dfs(self,root):
        self._dfs(root)
        return self.best_score, self.best_node

    def _dfs(self,n):
        # print("deeper::",output_lang.index2word[n.wordid.item()])
        # print("deeper::",output_lang.index2word[n.wordid.item()])
        # print(n.leng)
        # wait = input("PRESS ENTER TO CONTINUE.")
        # print("DEEPER")
        # print("serching ====>", '{:<30}'.format(str(n.currentList)))
        # print("serching ====>", "    score : ", "{:.2f}".format(n.score) )
        # print("serching ====>", "  thresh : ", "{:.2f}".format(self.best_score.item()))
        # print("")
        if n.wordid.item() == EOS_token or n.leng == MAX_LENGTH:
            return n.score, n
        decoder_input = n.wordid
        decoder_hidden = n.h
        decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, self.encoder_outputs)
        # for index in range
        #     print(decoder_output[0][index])
        topv, topi = decoder_output.topk(len(decoder_output[0]))
        # for new_k in range(len(decoder_output[0])):
        #     if topi[0][new_k].squeeze().item() < 10:
        #         print("testing!!-----------------")
        #         print(topi[0][new_k].squeeze().item())
        #         print(-topv[0][new_k])
        # wait = input("PRESS ENTER TO CONTINUE.")

    
        # if self.best_score <= n.score-torch.exp(decoder_output[0][EOS_token]):
        #     print()
        #     print("Original thresh_score ====>  ",self.best_score)
            
        #     currentList = n.currentList.copy()
        #     currentList.append(EOS_token)
        #     self.best_score = n.score-torch.exp(decoder_output[0][EOS_token])
        #     endnode = BeamSearchNode(decoder_hidden, n, currentList, torch.tensor(EOS_token), self.best_score, n.leng + 1)
        #     self.best_node = endnode
        #     print("Updating thresh_score ====>  ",self.best_score)
        #     print(self.best_node.currentList)

        EOS_token_score = n.score+(decoder_output[0][EOS_token])
        if EOS_token_score >=  self.best_score:
            # print("EOS bigger")
            # print("serching ====>", "  EOS_token_score : ", "{:.2f}".format(EOS_token_score.item()))
            # print("serching ====>", "  thresh : ", "{:.2f}".format(self.best_score.item()))
            # print("Original thresh_score ====>  ",self.best_score)
            # self.second_node = self.best_node
            
            currentList = n.currentList.copy()
            currentList.append(EOS_token)
            self.best_score = EOS_token_score
            endnode = BeamSearchNode(decoder_hidden, n, currentList, torch.tensor(EOS_token), self.best_score, n.leng + 1)
            self.best_node = endnode
            # print("serching ====>", '{:<30}'.format(str(self.best_node.currentList)) , "  score : ", "{:.2f}".format(self.best_score.item()))
    
    
        for new_k in range(len(decoder_output[0])):

    
            decoded_t = topi[0][new_k].squeeze().detach()
            score = topv[0][new_k]
            currentList = n.currentList.copy()
            currentList.append(decoded_t.item())
            new_score = n.score + score
            if new_score < self.best_score:
                # print("No better result Further")
                # print(new_score.item())
                # print(self.best_score.item())
                return self.best_score, self.best_node



            node = BeamSearchNode(decoder_hidden, n, currentList, decoded_t, new_score, n.leng + 1)
            # print("\rserching ====>", '{:<30}'.format(str(node.currentList)) ,"    score : ", "{:.2f}".format(node.score.item()) , "  thresh : ", "{:.2f}".format(self.best_score.item()),end="",flush=True)
            endscore, endnode = self._dfs(node)

    
    
            # print("new_score",new_score)
            # print("thresh_score",thresh_score)
            # wait = input("PRESS ENTER TO CONTINUE.")
            if endscore > self.best_score:
                
                # print("endnode--->", output_lang.index2word[endnode.wordid.item()])
                # print("endscore--->", endscore)
                # print("!!!!better result Further")
                # print("Original thresh_score ====>  ",self.best_score)
                self.best_score = endscore
                self.best_node = endnode
                # print("Updating thresh_score ====>  ",self.best_score)
                # if self.best_node != None:
                #     print(self.best_node.currentList)

        return self.best_score, self.best_node
"""The files are all in Unicode, to simplify we will turn Unicode
characters to ASCII, make everything lowercase, and trim most
punctuation.
"""

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

"""To read the data file we will split the file into lines, and then split
lines into pairs. The files are all English → Other Language, so if we
want to translate from Other Language → English I added the ``reverse``
flag to reverse the pairs.
"""
def readLangs(lang1, lang2, reverse=False, dataset = 'europarl-v7'):
    print("Reading lines...")

    if dataset == 'europarl-v7':

        # Read the file and split into lines
        lines = open('data/training/europarl-v7.%s-%s.%s' % (lang1, lang2,lang2), encoding='utf-8').\
            read().strip().split('\n')
        lines2 = open('data/training/europarl-v7.%s-%s.%s' % (lang1, lang2,lang1), encoding='utf-8').\
            read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = [[normalizeString(lines[item]),normalizeString(lines2[item])] for item in range(len(lines))]


    # if dataset == 'train':

    #     # Read the file and split into lines
    #     lines = open('0/out-train.%s-%s.%s' % (lang1, lang2,lang2), encoding='utf-8').\
    #         read().strip().split('\n')
    #     lines2 = open('0/out-train.%s-%s.%s' % (lang1, lang2,lang1), encoding='utf-8').\
    #         read().strip().split('\n')

    #     # Split every line into pairs and normalize
    #     pairs = [[normalizeString(lines[item]),normalizeString(lines2[item])] for item in range(len(lines))]
    # if dataset == 'val':

    #     # Read the file and split into lines
    #     lines = open('0/out-val.%s-%s.%s' % (lang1, lang2,lang2), encoding='utf-8').\
    #         read().strip().split('\n')
    #     lines2 = open('0/out-val.%s-%s.%s' % (lang1, lang2,lang1), encoding='utf-8').\
    #         read().strip().split('\n')

    #     # Split every line into pairs and normalize
    #     pairs = [[normalizeString(lines[item]),normalizeString(lines2[item])] for item in range(len(lines))]

    # if dataset == 'test':

    #     # Read the file and split into lines
    #     lines = open('0/out-test.%s-%s.%s' % (lang1, lang2,lang2), encoding='utf-8').\
    #         read().strip().split('\n')
    #     lines2 = open('0/out-test.%s-%s.%s' % (lang1, lang2,lang1), encoding='utf-8').\
    #         read().strip().split('\n')

    #     # Split every line into pairs and normalize
    #     pairs = [[normalizeString(lines[item]),normalizeString(lines2[item])] for item in range(len(lines))]
    else:
        # Read the file and split into lines
        lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
            read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instancess
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs
def readTestdata(lang1, lang2, reverse=False, dataset = 'train'):
    print("Reading lines...")

    if dataset == 'train':

        # Read the file and split into lines
        lines = open('0/out-train.%s-%s.%s' % (lang1, lang2,lang2), encoding='utf-8').\
            read().strip().split('\n')
        lines2 = open('0/out-train.%s-%s.%s' % (lang1, lang2,lang1), encoding='utf-8').\
            read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = [[normalizeString(lines[item]),normalizeString(lines2[item])] for item in range(len(lines))]
    if dataset == 'val':

        # Read the file and split into lines
        lines = open('0/out-val.%s-%s.%s' % (lang1, lang2,lang2), encoding='utf-8').\
            read().strip().split('\n')
        lines2 = open('0/out-val.%s-%s.%s' % (lang1, lang2,lang1), encoding='utf-8').\
            read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = [[normalizeString(lines[item]),normalizeString(lines2[item])] for item in range(len(lines))]

    if dataset == 'test':

        # Read the file and split into lines
        lines = open('0/out-test.%s-%s.%s' % (lang1, lang2,lang2), encoding='utf-8').\
            read().strip().split('\n')
        lines2 = open('0/out-test.%s-%s.%s' % (lang1, lang2,lang1), encoding='utf-8').\
            read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = [[normalizeString(lines[item]),normalizeString(lines2[item])] for item in range(len(lines))]
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
    return pairs
"""Since there are a *lot* of example sentences and we want to train
something quickly, we'll trim the data set to only relatively short and
simple sentences. Here the maximum length is 10 words (that includes
ending punctuation) and we're filtering to sentences that translate to
the form "I am" or "He is" etc. (accounting for apostrophes replaced
earlier).
"""

MAX_LENGTH = 10
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

# def filterPair(p):
#     return len(p[0].split(' ')) < MAX_LENGTH and \
#         len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

"""The full process for preparing the data is:

-  Read text file and split into lines, split lines into pairs
-  Normalize text, filter by length and content
-  Make word lists from sentences in pairs
"""


def prepareData(lang1, lang2, reverse=False,dataset='europarl-v7'):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse,dataset)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

# input_lang, output_lang, pairs = prepareData('fr', 'en', True)
input_lang, output_lang, pairs = prepareData('eng', 'fra', True,'original')
def partition (list_in, n):
    random.shuffle(list_in)
    return list_in[:int(n*len(list_in))],list_in[int(n*len(list_in)):]

# input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
# pairs, s_pairs = partition(pairs,0.8)
# v_pairs, T_pairs = partition(s_pairs,0.5)
# print(random.choice(pairs))
# print(random.choice(pairs))
# newpairs = []
# for n in range(10):
#     newpairs.append(random.choice(pairs))
# pairs = newpairs


"""The Seq2Seq Model
=================

A Recurrent Neural Network, or RNN, is a network that operates on a
sequence and uses its own output as input for subsequent steps.

A `Sequence to Sequence network <https://arxiv.org/abs/1409.3215>`__, or
seq2seq network, or `Encoder Decoder
network <https://arxiv.org/pdf/1406.1078v3.pdf>`__, is a model
consisting of two RNNs called the encoder and decoder. The encoder reads
an input sequence and outputs a single vector, and the decoder reads
that vector to produce an output sequence.

.. figure:: /_static/img/seq-seq-images/seq2seq.png
   :alt:

Unlike sequence prediction with a single RNN, where every input
corresponds to an output, the seq2seq model frees us from sequence
length and order, which makes it ideal for translation between two
languages.

Consider the sentence "Je ne suis pas le chat noir" → "I am not the
black cat". Most of the words in the input sentence have a direct
translation in the output sentence, but are in slightly different
orders, e.g. "chat noir" and "black cat". Because of the "ne/pas"
construction there is also one more word in the input sentence. It would
be difficult to produce a correct translation directly from the sequence
of input words.

With a seq2seq model the encoder creates a single vector which, in the
ideal case, encodes the "meaning" of the input sequence into a single
vector — a single point in some N dimensional space of sentences.

The Encoder
-----------

The encoder of a seq2seq network is a RNN that outputs some value for
every word from the input sentence. For every input word the encoder
outputs a vector and a hidden state, and uses the hidden state for the
next input word.

.. figure:: /_static/img/seq-seq-images/encoder-network.png
   :alt:
"""

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

"""The Decoder
-----------

The decoder is another RNN that takes the encoder output vector(s) and
outputs a sequence of words to create the translation.

Simple Decoder
^^^^^^^^^^^^^^

In the simplest seq2seq decoder we use only last output of the encoder.
This last output is sometimes called the *context vector* as it encodes
context from the entire sequence. This context vector is used as the
initial hidden state of the decoder.

At every step of decoding, the decoder is given an input token and
hidden state. The initial input token is the start-of-string ``<SOS>``
token, and the first hidden state is the context vector (the encoder's
last hidden state).

.. figure:: /_static/img/seq-seq-images/decoder-network.png
   :alt:
"""

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

"""I encourage you to train and observe the results of this model, but to
save space we'll be going straight for the gold and introducing the
Attention Mechanism.

Attention Decoder
^^^^^^^^^^^^^^^^^

If only the context vector is passed betweeen the encoder and decoder,
that single vector carries the burden of encoding the entire sentence.

Attention allows the decoder network to "focus" on a different part of
the encoder's outputs for every step of the decoder's own outputs. First
we calculate a set of *attention weights*. These will be multiplied by
the encoder output vectors to create a weighted combination. The result
(called ``attn_applied`` in the code) should contain information about
that specific part of the input sequence, and thus help the decoder
choose the right output words.

.. figure:: https://i.imgur.com/1152PYf.png
   :alt:

Calculating the attention weights is done with another feed-forward
layer ``attn``, using the decoder's input and hidden state as inputs.
Because there are sentences of all sizes in the training data, to
actually create and train this layer we have to choose a maximum
sentence length (input length, for encoder outputs) that it can apply
to. Sentences of the maximum length will use all the attention weights,
while shorter sentences will only use the first few.

.. figure:: /_static/img/seq-seq-images/attention-decoder-network.png
   :alt:
"""

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # self.dropout = nn.Dropout(self.dropout_p) 
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded) # should not use if we want argmax

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = self.out(output[0])
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



class DecoderRNN_log(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN_log, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)




class AttnDecoderRNN_log(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN_log, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        # embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)





"""Training the Model
------------------

To train we run the input sentence through the encoder, and keep track
of every output and the latest hidden state. Then the decoder is given
the ``<SOS>`` token as its first input, and the last hidden state of the
encoder as its first hidden state.

"Teacher forcing" is the concept of using the real target outputs as
each next input, instead of using the decoder's guess as the next input.
Using teacher forcing causes it to converge faster but `when the trained
network is exploited, it may exhibit
instability <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.378.4095&rep=rep1&type=pdf>`__.

You can observe outputs of teacher-forced networks that read with
coherent grammar but wander far from the correct translation -
intuitively it has learned to represent the output grammar and can "pick
up" the meaning once the teacher tells it the first few words, but it
has not properly learned how to create the sentence from the translation
in the first place.

Because of the freedom PyTorch's autograd gives us, we can randomly
choose to use teacher forcing or not with a simple if statement. Turn
``teacher_forcing_ratio`` up to use more of it.
"""

teacher_forcing_ratio = 0.3


def train(current_pair,input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()


    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    golden_score = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]# different setting in evaluation code
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    golden_decode = True
    if golden_decode:
        # print("feed the goden label:")
        # feed the golden label
        for di in range(target_length):
            # print(decoder_input.item())
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = target_tensor[di]  # feed the golden label
            golden_score += -torch.exp(decoder_output[0][target_tensor[di]])
        # print("golden-label : ", golden_score.item())
        # print(target_tensor.squeeze().tolist())
        # for item in target_tensor.squeeze().tolist():
        #     print(output_lang.index2word[item],end =" ")
        # print(golden_score.requires_grad)
        # print(golden_score.grad_fn)
    golden_score = golden_score
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden


    use_teacher_forcing = False if random.random() < teacher_forcing_ratio else False

    beam_search = True
    currentList = [decoder_input.item()]
    node = BeamSearchNode(decoder_hidden, None, currentList,  decoder_input, 0, 0)
    beam = 50
    nodes = PriorityQueue()
    nextnodes = PriorityQueue()
    endnodes = PriorityQueue()
    # start the queue
    nodes.put((-node.eval(), node))
    # f = open('test-output.txt', 'w')
    best_score = 0

    current_step = 0
    if use_teacher_forcing==False:

        # print("not use teacher-forced")
        # decoder_input = torch.tensor([[SOS_token]], device=device)

        # decoder_hidden = encoder_hidden

        # currentList = [decoder_input.item()]

        # node = BeamSearchNode(decoder_hidden, None, currentList,  decoder_input, 0, 0)
        # thresh_score = golden_score.item()

        # #exact_decode

        # score, n = DFS(thresh_score,encoder_outputs,decoder).dfs(node)

        # endnodes.put((-(n.eval()),n))







################################################ BEAM SEARCH ######################################################
        while True:
            beam =50
            for _ in range(nodes.qsize()):
                S, n = nodes.get()
                if n.leng > MAX_LENGTH:
                    endnodes.put((-(n.eval()),n))
                    # print("Enough!")
                    if endnodes.qsize() == beam:
                        break
                    continue
                if n.wordid.item() == EOS_token or n.leng == target_length:
                    endnodes.put((-(n.eval()),n))
                    # print("got it")
                    if endnodes.qsize() == beam:
                        break
                    continue
                decoder_input = n.wordid
                decoder_hidden = n.h
                # print(decoder_input)
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = (-decoder_output).topk(beam)
                for new_k in range(beam):
                    # print("new_k",new_k)
                    decoded_t = topi[0][new_k].squeeze().detach()
                    score = -torch.exp(-topv[0][new_k])
                    currentList = n.currentList.copy()
                    currentList.append(decoded_t.item())
                    new_score = n.score +score
                    node = BeamSearchNode(decoder_hidden, n, currentList, decoded_t, new_score, n.leng + 1)
                    nextnodes.put((-(node.eval()),node))
            if endnodes.qsize() == beam:
                break
            while not nodes.empty():
                nodes.get() 
            if nextnodes.empty():
                return 0
            for _ in range(beam):
                nodes.put((nextnodes.get()))
            while not nextnodes.empty():
                nextnodes.get()
##################################################################################################################


    # ref = []
    # ref.append('SOS')
    # for item in current_pair[1].split():
    #     ref.append(item)
    # ref.append('EOS')
    score, n = endnodes.get()
    # candidates = []
    # candidates_score = []
    # for top in range(endnodes.qsize()):
    #     score, n = endnodes.get()
    #     candidates_score.append(score)
            
        # utterance = []
        # utterance.append(output_lang.index2word[n.wordid.item()])
        # while n.prevNode != None:
        #     n = n.prevNode
        #     utterance.append(output_lang.index2word[n.wordid.item()])
        
        # utterance = utterance[::-1]
        # candidates.append(utterance)

    # best_index = 0
    # best_bleu = 0
    # for i in range(len(candidates)):
    #     current_bleu = sentence_bleu([ref],candidates[i])
    #     if best_bleu < current_bleu:
    #         best_bleu = current_bleu
    #         best_index = i


    loss = -golden_score
    # print([candidates[0]])
    # print(ref)
    # print([candidates[best_index]])
    # loss = candidates_score[best_index]
    # print(candidates_score)
    loss += -score
    # print(loss.requires_grad)
    # print(loss.grad_fn)  
    # print(loss.item())


    TFscore = 0

    if loss <0:
        for di in range(target_length):
            # print(decoder_input.item())
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = target_tensor[di]  # feed the golden label
            topv, topi = (-decoder_output).topk(1)
            TFscore += -torch.exp(-topv.squeeze().detach())
        loss = -golden_score -TFscore


    print(loss.item())
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.5)
    # torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.5)
    encoder_optimizer.step()
    decoder_optimizer.step()

    # print("===========================")
    return loss.item()


def val_train(current_pair,input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()


    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    golden_score = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]# different setting in evaluation code
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    golden_decode = True
    if golden_decode:
        # feed the golden label
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = target_tensor[di]  # feed the golden label
            golden_score += -torch.exp(decoder_output[0][target_tensor[di]])
        # print("golden-label : ", golden_score.item())
        # print(target_tensor.squeeze().tolist())
        # for item in target_tensor.squeeze().tolist():
        #     print(output_lang.index2word[item],end =" ")
        # print(golden_score.requires_grad)
        # print(golden_score.grad_fn)
    golden_score = golden_score
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    

    beam_search = True
    currentList = [decoder_input.item()]
    node = BeamSearchNode(decoder_hidden, None, currentList,  decoder_input, 0, 0)
    beam = 5
    nodes = PriorityQueue()
    nextnodes = PriorityQueue()
    endnodes = PriorityQueue()
    # start the queue
    nodes.put((-node.eval(), node))
    # f = open('test-output.txt', 'w')
    best_score = 0
    if beam_search:
        while True:   
            for _ in range(nodes.qsize()):
                S, n = nodes.get()
                if n.leng > MAX_LENGTH:
                    endnodes.put((-(n.eval()),n))
                    # print("Enough!")
                    if endnodes.qsize() == beam:
                        break
                    continue
                if n.wordid.item() == EOS_token:
                    endnodes.put((-(n.eval()),n))
                    # print("got it")
                    if endnodes.qsize() == beam:
                        break
                    continue
                decoder_input = n.wordid
                decoder_hidden = n.h
                # print(decoder_input)
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = (-decoder_output).topk(beam)
                for new_k in range(beam):
                    # print("new_k",new_k)
                    decoded_t = topi[0][new_k].squeeze().detach()
                    score = -torch.exp(-topv[0][new_k])
                    currentList = n.currentList.copy()
                    currentList.append(decoded_t.item())
                    new_score = n.score +score
                    node = BeamSearchNode(decoder_hidden, n, currentList, decoded_t, new_score, n.leng + 1)
                    nextnodes.put((-(node.eval()),node))
            if endnodes.qsize() == beam:
                break
            while not nodes.empty():
                nodes.get() 
            if nextnodes.empty():
                return 0
            for _ in range(beam):
                nodes.put((nextnodes.get()))
            while not nextnodes.empty():
                nextnodes.get() 
        # print("endnodes.qsize()", endnodes.qsize())
        # for top in range(endnodes.qsize()):
        #     score, n = endnodes.get()
        #     best_score = -golden_score
        #     if top == 0:
        #         best_score = score
            # utterance = []
            # utterance.append(output_lang.index2word[n.wordid.item()])
            # while n.prevNode != None:
            #     n = n.prevNode
            #     utterance.append(output_lang.index2word[n.wordid.item()])
            # utterance = utterance[::-1]
            # print(utterance)
    # print([current_pair[1]])
    ref = []
    ref.append('SOS')
    for item in current_pair[1].split():
        ref.append(item)
    ref.append('EOS')

    candidates = []
    candidates_score = []
    for top in range(endnodes.qsize()):
        score, n = endnodes.get()
        candidates_score.append(score)
            
        utterance = []
        utterance.append(output_lang.index2word[n.wordid.item()])
        while n.prevNode != None:
            n = n.prevNode
            utterance.append(output_lang.index2word[n.wordid.item()])
        
        utterance = utterance[::-1]
        candidates.append(utterance)

    best_index = 0
    best_bleu = 0
    for i in range(len(candidates)):
        current_bleu = sentence_bleu([ref],candidates[i])
        if best_bleu < current_bleu:
            best_bleu = current_bleu
            best_index = i


    loss = -golden_score
    # print([candidates[0]])
    # print([candidates[best_index]])
    # loss = candidates_score[best_index]
    loss += -candidates_score[0]
    # print(loss.requires_grad)
    # print(loss.grad_fn)  

    # print()
    # print("search-label : ",-(best_score).item())
    # print("-------- loss ---------    >>>>>>>>>    ",loss.item(),"  <<<<<<<<<<")
    # wait = input("PRESS ENTER TO CONTINUE.")


    # # testing-----------------------------------------------
    # loss_test = 0

    # decoder_input = torch.tensor([[SOS_token]], device=device)

    # decoder_hidden = encoder_hidden

    # golden_decode = True
    # if golden_decode:
    #     # feed the golden label
    #     for di in range(target_length):
    #         decoder_output, decoder_hidden, decoder_attention = decoder(
    #             decoder_input, decoder_hidden, encoder_outputs)

    #         decoder_input = target_tensor[di]  # feed the golden label
    #         loss_test += (decoder_output[0][target_tensor[di]])
    #     print("-------test-golden-label-test---------")
    #     print(loss_test)
    #     print(target_tensor) 



    cost = 1.0 - sentence_bleu([ref],candidates[0])
    # print(cost)
    return loss, cost
"""This is a helper function to print time elapsed and estimated time
remaining given the current time and progress %.
"""
def train_log(current_pair,input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else True

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()

def val_train_log(current_pair,input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()


    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = False
    decoded_words = []
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])


    ref = []
    for item in current_pair[1].split():
        ref.append(item)
    current_bleu = sentence_bleu([ref],decoded_words)

    cost = 1 - current_bleu
    return loss, cost

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

"""The whole training process looks like this:

-  Start a timer
-  Initialize optimizers and criterion
-  Create set of training pairs
-  Start empty losses array for plotting

Then we call ``train`` many times and occasionally print the progress (%
of examples, time so far, estimated time) and average loss.
"""

def trainIters(pairs, v_pairs, T_pairs,encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01, num = 0):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    val_loss_pre = float('inf')
    val_cost_pre = float('inf')

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate,momentum=0.5)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate,momentum=0.5)
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, 1.0, gamma=0.95)
    decoder_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, 1.0, gamma=0.95)
    randomNUM = np.random.permutation(n_iters)
    # print(pairs)
    current_pairs = [pairs[randomNUM[i]%(len(pairs))] for i in range(n_iters)]
    training_pairs = [tensorsFromPair(current_pairs[i])
                      for i in range(n_iters)]
    val_current_pairs = v_pairs
    T_current_pairs = T_pairs
    val_pairs = [tensorsFromPair(val_current_pairs[i])
                        for i in range(len(v_pairs))]
    T_pairs = [tensorsFromPair(T_current_pairs[i])
                        for i in range(len(T_pairs))]         

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]


        loss = train(current_pairs[iter - 1],input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss




        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0


            val_loss = 0
            val_cost = 0

            for item in range(len(val_pairs)):
                val_pair = val_pairs[item]
                input_tensor = val_pair[0]
                target_tensor = val_pair[1]

                loss, cost = val_train(val_current_pairs[item],input_tensor, target_tensor, encoder,decoder, encoder_optimizer, decoder_optimizer, criterion)
                val_loss += loss
                val_cost += cost

            val_loss = val_loss/len(val_pairs)

            # if (val_loss_pre < val_loss) and (val_cost_pre < val_cost):
            #     print("**********************Congraduation!!!!!********************")
            #     break

            # val_loss_pre = val_loss
            # val_cost_pre = val_cost

            T_loss = 0
            T_cost = 0

            # for item in range(len(T_pairs)):
            #     T_pair = T_pairs[item]
            #     input_tensor = T_pair[0]
            #     target_tensor = T_pair[1]

            #     loss, cost = val_train(T_current_pairs[item],input_tensor, target_tensor, encoder,decoder, encoder_optimizer, decoder_optimizer, criterion)
            #     T_loss += loss
            #     T_cost += cost

            T_loss = T_loss/len(T_pairs)
            encoder_scheduler.step()
            decoder_scheduler.step()
            torch.save(encoder1.state_dict(),'{0}/NOTEACHpercep_encoder1__{1}_{2}'.format( num, val_loss.item(), val_cost))
            torch.save(attn_decoder1.state_dict(),'{0}/NOTEACHpercep_attn_decoder1__{1}_{2}'.format( num, val_loss.item(), val_cost ))
            # torch.save(encoder1.state_dict(),'{0}/TRUEONLYTEACHpercep_encoder1__{1}_{2}_{3}_{4}'.format( num, val_loss.item(), val_cost,T_loss.item(), T_cost))
            # torch.save(attn_decoder1.state_dict(),'{0}/TRUEONLYTEACHpercep_attn_decoder1__{1}_{2}_{3}_{4}'.format( num, val_loss.item(), val_cost,T_loss.item(), T_cost ))
            print('%s (%d %d%%) %.4f  %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg, val_loss))



def trainIters_log(pairs, v_pairs, T_pairs,encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01,  num = 0):
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    val_loss_pre = float('inf')
    val_cost_pre = float('inf')

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    randomNUM = np.random.permutation(n_iters)

    current_pairs = [pairs[randomNUM[i]%(len(pairs))] for i in range(n_iters)]
    training_pairs = [tensorsFromPair(current_pairs[i])
                      for i in range(n_iters)]

    val_current_pairs = v_pairs
    T_current_pairs = T_pairs
    val_pairs = [tensorsFromPair(val_current_pairs[i])
                        for i in range(len(v_pairs))]
    T_pairs = [tensorsFromPair(T_current_pairs[i])
                        for i in range(len(T_pairs))]         
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train_log(current_pairs[iter - 1],input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss


        



        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            val_loss = 0
            val_cost = 0

            for item in range(len(val_pairs)):
                val_pair = val_pairs[item]
                input_tensor = val_pair[0]
                target_tensor = val_pair[1]

                loss, cost = val_train_log(val_current_pairs[item],input_tensor, target_tensor, encoder,decoder, encoder_optimizer, decoder_optimizer, criterion)
                val_loss += loss
                val_cost += cost

            val_loss = val_loss/len(val_pairs)

            # if (val_loss_pre < val_loss) and (val_cost_pre < val_cost):
            #     print("**********************Congraduation!!!!!********************")
            #     break


            T_loss = 0
            T_cost = 0

            for item in range(len(T_pairs)):
                T_pair = T_pairs[item]
                input_tensor = T_pair[0]
                target_tensor = T_pair[1]

                loss, cost = val_train_log(T_current_pairs[item],input_tensor, target_tensor, encoder,decoder, encoder_optimizer, decoder_optimizer, criterion)
                T_loss += loss
                T_cost += cost

            T_loss = T_loss/len(T_pairs)



            torch.save(encoder1.state_dict(),'{0}/ONLYTEACH-pro_encoder1__{1}_{2}_{3}_{4}_{5}'.format( num, val_loss.item(), val_cost,T_loss.item(), T_cost,iter / n_iters * 100))
            torch.save(attn_decoder1.state_dict(),'{0}/ONLYTEACH-pro_attn_decoder1__{1}_{2}_{3}_{4}_{5}'.format( num, val_loss.item(), val_cost,T_loss.item(), T_cost,iter / n_iters * 100 ))
            
            outfile = open('{0}/ONLYTEACH-pro_encoder1__{1}'.format( num,iter / n_iters * 100), 'w')
            orig_stdout = sys.stdout
            sys.stdout = outfile
            evaluateRandomly_log(encoder1, attn_decoder1,100)
            outfile.close()

            outfile_beam = open('{0}/Beam_ONLYTEACH-pro_encoder1__{1}'.format( num,iter / n_iters * 100), 'w')
            sys.stdout = outfile_beam
            evaluateRandomly_log_Beam(encoder1, attn_decoder1,100,beam=5)
            outfile_beam.close()

            outfile_beam10 = open('{0}/Beam200_ONLYTEACH-pro_encoder1__{1}'.format( num,iter / n_iters * 100), 'w')
            sys.stdout = outfile_beam10
            evaluateRandomly_log_Beam(encoder1, attn_decoder1,100,beam=200)
            outfile_beam10.close()

            outfile_dfs = open('{0}/Dfs_ONLYTEACH-pro_encoder1__{1}'.format( num,iter / n_iters * 100), 'w')
            sys.stdout = outfile_dfs
            evaluateRandomly_log_Dfs(encoder1, attn_decoder1,100,time=5)
            outfile_dfs.close()

            # outfile_dfs10 = open('{0}/Dfs10_ONLYTEACH-pro_encoder1__{1}'.format( num,iter / n_iters * 100), 'w')
            # sys.stdout = outfile_dfs10
            # evaluateRandomly_log_Dfs(encoder1, attn_decoder1,100,time=500)
            # outfile_dfs10.close()



            os.system("cat {0}/ONLYTEACH-pro_encoder1__{1} | sacrebleu 0/out-test.fr-en.en".format( num,iter / n_iters * 100))
            os.system("cat {0}/Beam_ONLYTEACH-pro_encoder1__{1} | sacrebleu 0/out-test.fr-en.en".format( num,iter / n_iters * 100))
            os.system("cat {0}/Beam200_ONLYTEACH-pro_encoder1__{1} | sacrebleu 0/out-test.fr-en.en".format( num,iter / n_iters * 100))
            os.system("cat {0}/Dfs_ONLYTEACH-pro_encoder1__{1} | sacrebleu 0/out-test.fr-en.en".format( num,iter / n_iters * 100))
            # os.system("cat {0}/Dfs10_ONLYTEACH-pro_encoder1__{1} | sacrebleu 0/out-test.fr-en.en".format( num,iter / n_iters * 100))
            sys.stdout = orig_stdout
            print('%s (%d %d%%) %.4f  %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg, val_loss))
"""Plotting results
----------------

Plotting is done with matplotlib, using the array of loss values
``plot_losses`` saved while training.
"""



# import matplotlib.pyplot as plt
# plt.switch_backend('agg')
# import matplotlib.ticker as ticker
# import numpy as np


# def showPlot(points):
#     plt.figure()
#     fig, ax = plt.subplots()
#     # this locator puts ticks at regular intervals
#     loc = ticker.MultipleLocator(base=0.2)
#     ax.yaxis.set_major_locator(loc)
#     plt.plot(points)

"""Evaluation
==========

Evaluation is mostly the same as training, but there are no targets so
we simply feed the decoder's predictions back to itself for each step.
Every time it predicts a word we add it to the output string, and if it
predicts the EOS token we stop there. We also store the decoder's
attention outputs for display later.
"""

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden
        currentList = [decoder_input.item()]

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        
        node = BeamSearchNode(decoder_hidden, None, currentList,  decoder_input, 0, 0)
        beam = 12
        nodes = PriorityQueue()
        nextnodes = PriorityQueue()
        endnodes = PriorityQueue()


        # start the queue
        nodes.put((-node.eval(), node))

        # f = open('test-output.txt', 'w')
        beam_search = True
        best_score = 0


        while True:
                
            for _ in range(nodes.qsize()):

                S, n = nodes.get()
                if n.leng > MAX_LENGTH:
                    endnodes.put((-(n.eval()),n))
                    # print("Enough!")
                    if endnodes.qsize() == beam:
                        break
                    continue

                if n.wordid.item() == EOS_token:
                    endnodes.put((-(n.eval()),n))
                    # print("got it")
                    # print("!!!!!!!!!!!!!!!!!!!!!!!!serching ====>", '{:<30}'.format(str(n.currentList)) , "  score : ", "{:.2f}".format(n.score))
                    if endnodes.qsize() == beam:
                        break
                    continue

                decoder_input = n.wordid
                decoder_hidden = n.h
                # print("serching ====>", '{:<30}'.format(str(n.currentList)) , "  score : ", "{:.2f}".format(n.score))

                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = (-decoder_output).topk(beam)
                # print("serching ====>", '{:<30}'.format(str(n.currentList)+"EOS") , "  score : ", "{:.2f}".format(n.score-torch.exp(-topv[0][EOS_token])))
                for new_k in range(beam):
                    # print("new_k",new_k)
                    decoded_t = topi[0][new_k].squeeze().detach()
                    score = -torch.exp(-topv[0][new_k])
                    currentList = n.currentList.copy()
                    currentList.append(decoded_t.item())
                    new_score = n.score +score

                    node = BeamSearchNode(decoder_hidden, n, currentList, decoded_t, new_score, n.leng + 1)
                    # print("serching ====>", '{:<30}'.format(str(node.currentList)) , "  score : ", "{:.2f}".format(node.score))
                    nextnodes.put((-(node.eval()),node))
                
            if endnodes.qsize() == beam:
                break

            while not nodes.empty():
                nodes.get() 

            if nextnodes.empty():
                return 0

            for _ in range(beam):
                nodes.put((nextnodes.get()))

            while not nextnodes.empty():
                nextnodes.get() 
    
        # print("endnodes.qsize()", endnodes.qsize())

        candidates = []
        for top in range(endnodes.qsize()):
            score, n = endnodes.get()
            if top == 0:
                best_score = score
            
            utterance = []
            utterance.append(output_lang.index2word[n.wordid.item()])
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(output_lang.index2word[n.wordid.item()])
        
            utterance = utterance[::-1]
            candidates.append(utterance)

        return candidates

def evaluate_log(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(topv)
            if topi.item() == EOS_token:
                decoded_words.append('EOS')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words
        # return decoded_words, decoder_attentions[:di + 1]
def evaluate_log_Beam(encoder, decoder, sentence,beam=5, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden
        currentList = [decoder_input.item()]

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        
        node = BeamSearchNode(decoder_hidden, None, currentList,  decoder_input, 0, 0)
        nodes = PriorityQueue()
        nextnodes = PriorityQueue()
        endnodes = PriorityQueue()


        # start the queue
        nodes.put((-node.eval(), node))

        # f = open('test-output.txt', 'w')
        beam_search = True
        best_score = 0


        while True:
                
            for _ in range(nodes.qsize()):

                S, n = nodes.get()
                if n.leng == MAX_LENGTH or n.wordid.item() == EOS_token:
                    endnodes.put(((-n.eval()),n))
                    # print("Enough!")
                    if endnodes.qsize() == beam:
                        break
                    continue

                decoder_input = n.wordid
                decoder_hidden = n.h
                # print("serching ====>", '{:<30}'.format(str(n.currentList)) , "  score : ", "{:.2f}".format(n.score))

                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = (decoder_output).topk(beam)
                # print("serching ====>", '{:<30}'.format(str(n.currentList)+"EOS") , "  score : ", "{:.2f}".format(n.score-torch.exp(-topv[0][EOS_token])))
                for new_k in range(beam):
                    # print("new_k",new_k)
                    decoded_t = topi[0][new_k].squeeze().detach()
                    score = topv[0][new_k]
                    currentList = n.currentList.copy()
                    currentList.append(decoded_t.item())
                    new_score = n.score +score

                    node = BeamSearchNode(decoder_hidden, n, currentList, decoded_t, new_score, n.leng + 1)
                    # print("serching ====>", '{:<30}'.format(str(node.currentList)) , "  score : ", "{:.2f}".format(node.score))
                    nextnodes.put(((-node.eval()),node))
                
            if endnodes.qsize() == beam:
                break

            while not nodes.empty():
                nodes.get() 

            if nextnodes.empty():
                return 0

            for _ in range(beam):
                nodes.put((nextnodes.get()))

            while not nextnodes.empty():
                nextnodes.get() 
    
        # print("endnodes.qsize()", endnodes.qsize())

        # candidates = []
        # for top in range(endnodes.qsize()):
        #     score, n = endnodes.get()
        #     if top == 0:
        #         best_score = score
            
        #     utterance = []
        #     utterance.append(output_lang.index2word[n.wordid.item()])
        #     while n.prevNode != None:
        #         n = n.prevNode
        #         utterance.append(output_lang.index2word[n.wordid.item()])
        
        #     utterance = utterance[::-1]
        #     candidates.append(utterance)
        score, n = endnodes.get()
            
        utterance = []
        utterance.append(output_lang.index2word[n.wordid.item()])
        while n.prevNode != None:
            n = n.prevNode
            utterance.append(output_lang.index2word[n.wordid.item()])
        
        utterance = utterance[::-1]
        return utterance

def evaluate_log_Dfs(encoder, decoder, sentence,time =5, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden
        currentList = [decoder_input.item()]

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        
        node = BeamSearchNode(decoder_hidden, None, currentList,  decoder_input, 0, 0)





        dfsclass = DFS(-float("inf"),encoder_outputs,decoder)
        try:
            with time_limit(time):
                score, n = dfsclass.dfs(node)
        except TimeoutException as e:
            n = dfsclass.best_node
            score = dfsclass.best_score
            # print("OUT OF TIME LIMIT")

        # score, n = dfsclass.dfs(node)
        if n == None:
            # print("None result from DFS-……-")
            return ["none"]

        # print("")
        # print(str(n.currentList)+"score = "+str(score))
        utterance = []
        utterance.append(output_lang.index2word[n.wordid.item()])
        while n.prevNode != None:
            n = n.prevNode
            utterance.append(output_lang.index2word[n.wordid.item()])
        utterance = utterance[::-1]


        return utterance
"""We can evaluate random sentences from the training set and print out the
input, target, and output to make some subjective quality judgements:
"""
def listToString(s):  
    
    # initialize an empty string 
    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 += ele   
    
    # return string   
    return str1 

def evaluateRandomly(encoder, decoder, n=100):
    # score1 = 0
    # score2 = 0
    # score3 = 0
    # score4 = 0
    # score5 = 0
    for pair in T_pairs:
        # print('>', pair[0])
        # print('=', pair[1])
        # ref = []
        # ref.append('SOS')
        # for item in pair[1].split():
        #     ref.append(item)
        # ref.append('EOS')
        candidates = evaluate(encoder, decoder, pair[0])
        # for index in range(len(candidates)):
        #     print('<',candidates[index])
        # print([ref])
        output = candidates[0][1:-1]
        sentence = " ".join(output)
        print(sentence)
        # bleu1 = sentence_bleu([ref],candidates[0], weights=(1, 0, 0, 0))
        # bleu2 = sentence_bleu([ref],candidates[0], weights=(0.5, 0.5, 0, 0))
        # bleu3 = sentence_bleu([ref],candidates[0], weights=(0.33, 0.33, 0.33, 0))
        # bleu4 = sentence_bleu([ref],candidates[0], weights=(0.25, 0.25, 0.25, 0.25))
        # bleu = sentence_bleu([ref],candidates[0])
        # score1 += bleu1
        # score2 += bleu2
        # score3 += bleu3
        # score4 += bleu4
        # score5 += bleu
        # print(bleu1)
        # print(bleu2)
        # print(bleu3)
        # print(bleu4)
        # print(bleu)

    # print(score1/len(T_pairs))
    # print(score2/len(T_pairs))
    # print(score3/len(T_pairs))
    # print(score4/len(T_pairs))
    # print(score5/len(T_pairs))
    return 


def evaluateRandomly_log(encoder, decoder, n=100):
    # score1 = 0
    # score2 = 0
    # score3 = 0
    # score4 = 0
    # score5 = 0
    for pair in T_pairs:
        # print('>', pair[0])
        # print('=', pair[1])
        # ref = []
        # ref.append('SOS')
        # for item in pair[1].split():
        #     ref.append(item)
        # ref.append('EOS')
        output_words = evaluate_log(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words[:-1])
        # candidates = []
        # candidates.append('SOS')
        # for item in output_sentence.split():
        #     candidates.append(item)
        print(output_sentence)
        # print(candidates)
        # print([ref])
        # bleu1 = sentence_bleu([ref],candidates, weights=(1, 0, 0, 0))
        # bleu2 = sentence_bleu([ref],candidates, weights=(0.5, 0.5, 0, 0))
        # bleu3 = sentence_bleu([ref],candidates, weights=(0.33, 0.33, 0.33, 0))
        # bleu4 = sentence_bleu([ref],candidates, weights=(0.25, 0.25, 0.25, 0.25))
        # bleu = sentence_bleu([ref],candidates)
        # score1 += bleu1
        # score2 += bleu2
        # score3 += bleu3
        # score4 += bleu4
        # score5 += bleu
        # print(bleu1)
        # print(bleu2)
        # print(bleu3)
        # print(bleu4)
        # print(bleu)

    # print(score1/len(T_pairs))
    # print(score2/len(T_pairs))
    # print(score3/len(T_pairs))
    # print(score4/len(T_pairs))
    # print(score5/len(T_pairs))
    return 

def evaluateRandomly_log_Beam(encoder, decoder, n=100,beam=5):
    # score1 = 0
    # score2 = 0
    # score3 = 0
    # score4 = 0
    # score5 = 0
    for pair in T_pairs:
        # print('>', pair[0])
        # print('=', pair[1])
        # ref = []
        # ref.append('SOS')
        # for item in pair[1].split():
        #     ref.append(item)
        # ref.append('EOS')
        output_words = evaluate_log_Beam(encoder, decoder, pair[0],beam)
        output_sentence = ' '.join(output_words[1:-1])
        # candidates = []
        # candidates.append('SOS')
        # for item in output_sentence.split():
        #     candidates.append(item)
        print(output_sentence)
        # print(candidates)
        # print([ref])
        # bleu1 = sentence_bleu([ref],candidates, weights=(1, 0, 0, 0))
        # bleu2 = sentence_bleu([ref],candidates, weights=(0.5, 0.5, 0, 0))
        # bleu3 = sentence_bleu([ref],candidates, weights=(0.33, 0.33, 0.33, 0))
        # bleu4 = sentence_bleu([ref],candidates, weights=(0.25, 0.25, 0.25, 0.25))
        # bleu = sentence_bleu([ref],candidates)
        # score1 += bleu1
        # score2 += bleu2
        # score3 += bleu3
        # score4 += bleu4
        # score5 += bleu
        # print(bleu1)
        # print(bleu2)
        # print(bleu3)
        # print(bleu4)
        # print(bleu)

    # print(score1/len(T_pairs))
    # print(score2/len(T_pairs))
    # print(score3/len(T_pairs))
    # print(score4/len(T_pairs))
    # print(score5/len(T_pairs))
    return 
def evaluateRandomly_log_Dfs(encoder, decoder, n=100,time =5):
    # score1 = 0
    # score2 = 0
    # score3 = 0
    # score4 = 0
    # score5 = 0
    for pair in T_pairs:
        # print('>', pair[0])
        # print('=', pair[1])
        # ref = []
        # ref.append('SOS')
        # for item in pair[1].split():
        #     ref.append(item)
        # ref.append('EOS')
        output_words = evaluate_log_Dfs(encoder, decoder, pair[0],time)
        output_sentence = ' '.join(output_words[1:-1])
        # candidates = []
        # candidates.append('SOS')
        # for item in output_sentence.split():
        #     candidates.append(item)
        print(output_sentence)
        # print(candidates)
        # print([ref])
        # bleu1 = sentence_bleu([ref],candidates, weights=(1, 0, 0, 0))
        # bleu2 = sentence_bleu([ref],candidates, weights=(0.5, 0.5, 0, 0))
        # bleu3 = sentence_bleu([ref],candidates, weights=(0.33, 0.33, 0.33, 0))
        # bleu4 = sentence_bleu([ref],candidates, weights=(0.25, 0.25, 0.25, 0.25))
        # bleu = sentence_bleu([ref],candidates)
        # score1 += bleu1
        # score2 += bleu2
        # score3 += bleu3
        # score4 += bleu4
        # score5 += bleu
        # print(bleu1)
        # print(bleu2)
        # print(bleu3)
        # print(bleu4)
        # print(bleu)

    # print(score1/len(T_pairs))
    # print(score2/len(T_pairs))
    # print(score3/len(T_pairs))
    # print(score4/len(T_pairs))
    # print(score5/len(T_pairs))
    return 
"""Training and Evaluating
=======================

With all these helper functions in place (it looks like extra work, but
it makes it easier to run multiple experiments) we can actually
initialize a network and start training.

Remember that the input sentences were heavily filtered. For this small
dataset we can use relatively small networks of 256 hidden nodes and a
single GRU layer. After about 40 minutes on a MacBook CPU we'll get some
reasonable results.

.. Note::
   If you run this notebook you can train, interrupt the kernel,
   evaluate, and continue training later. Comment out the lines where the
   encoder and decoder are initialized and run ``trainIters`` again.
"""


hidden_size = 256

# for num in range(10):
#     orig_stdout = sys.stdout
#     pairs, s_pairs = partition(pairs,0.8)
#     v_pairs, T_pairs = partition(s_pairs,0.5)

#     if not os.path.exists(str(num)):
#         os.makedirs(str(num))

#     f00 = open('%d/out-train.fr-en.fr' % num, 'w')
#     sys.stdout = f00
#     for wpairs in pairs:
#         print(wpairs[0])
#     f00.close()

#     f01 = open('%d/out-train.fr-en.en' % num, 'w')
#     sys.stdout = f01
#     for wpairs in pairs:
#         print(wpairs[1])
#     f01.close()


#     f1 = open('%d/out-val.fr-en.fr' % num, 'w')
#     sys.stdout = f1
#     for wpairs in v_pairs:
#         print(wpairs[0])
#     f1.close()

#     f2 = open('%d/out-val.fr-en.en' % num, 'w')
#     sys.stdout = f2
#     for wpairs in v_pairs:
#         print(wpairs[1])
#     f2.close()

#     f3 = open('%d/out-test.fr-en.fr' % num, 'w')
#     sys.stdout = f3
#     for wpairs in T_pairs:
#         print(wpairs[0])
#     f3.close()

#     f4 = open('%d/out-test.fr-en.en' % num, 'w')
#     sys.stdout = f4
#     for wpairs in T_pairs:
#         print(wpairs[1])
#     f4.close()


#     f_out = open('%d/out-log.txt' % num, 'w')
#     sys.stdout = f_out
#     # sys.stdout = orig_stdout

#     encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
#     attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
#     # encoder1.load_state_dict(torch.load("data/model/beam1/encoder1"))
#     # attn_decoder1.load_state_dict(torch.load('data/model/beam1/attn_decoder1'))
#     # encoder1.load_state_dict(torch.load("data/model/percep_encoder1_0_-0.03336292505264282_881.9394792030312_-0.029477035626769066_871.6695669618281"))
#     # attn_decoder1.load_state_dict(torch.load('data/model/percep_attn_decoder1_0_-0.03336292505264282_881.9394792030312_-0.029477035626769066_871.6695669618281'))
#     # encoder1.load_state_dict(torch.load("data/model/percep_encoder1"))
#     # attn_decoder1.load_state_dict(torch.load('data/model/percep_attn_decoder1'))
#     trainIters(pairs, v_pairs, T_pairs, encoder1, attn_decoder1, 10*len(pairs), print_every=len(pairs), learning_rate=0.01, num = num)
#     # evaluateRandomly(encoder1, attn_decoder1,100)


#     encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
#     attn_decoder1 = AttnDecoderRNN_log(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
#     encoder1.load_state_dict(torch.load("data/model/pro_encoder1_0_13.57414722442627_746.3501116239448_13.328173637390137_728.678738916424"))
#     attn_decoder1.load_state_dict(torch.load('data/model/percep_attn_decoder1_0_13.57414722442627_746.3501116239448_13.328173637390137_728.678738916424'))

#     # encoder1.load_state_dict(torch.load("data/model/pro_encoder1"))
#     # attn_decoder1.load_state_dict(torch.load('data/model/pro_attn_decoder1'))
#     trainIters_log(pairs, v_pairs, T_pairs,encoder1, attn_decoder1, 10*len(pairs), print_every=len(pairs), learning_rate=0.01, num = num)
#     # evaluateRandomly_log(encoder1, attn_decoder1,100)



#     sys.stdout = orig_stdout
#     f_out.close()
orig_stdout = sys.stdout





pairs = readTestdata('fr', 'en', True, 'train')
v_pairs = readTestdata('fr', 'en', True, 'val')
T_pairs = readTestdata('fr', 'en', True, 'test')
lr = 0.01
# encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
# # encoder1.load_state_dict(torch.load("0/TRUEONLYTEACHpercep_encoder1__0.17780013382434845_714.9898776134119_0.14045269787311554_699.7516417223444"))
# # attn_decoder1.load_state_dict(torch.load('0/TRUEONLYTEACHpercep_attn_decoder1__0.17780013382434845_714.9898776134119_0.14045269787311554_699.7516417223444'))
# # encoder1.load_state_dict(torch.load("0/percep_encoder1__0.0233584214001894_711.356986880004_-0.07643373310565948_691.570443207043"))
# # attn_decoder1.load_state_dict(torch.load("0/percep_attn_decoder1__0.0233584214001894_711.356986880004_-0.07643373310565948_691.570443207043"))

# # encoder1.load_state_dict(torch.load("0/TRUEONLYTEACHpercep_encoder1__0.19646424055099487_789.6468949012324_0.1551685780286789_768.180439672762"))
# # attn_decoder1.load_state_dict(torch.load("0/TRUEONLYTEACHpercep_attn_decoder1__0.19646424055099487_789.6468949012324_0.1551685780286789_768.180439672762"))

# # encoder1.load_state_dict(torch.load("0/2TEACHpercep_encoder1__0.18038101494312286_715.1640927611335_0.1423390507698059_699.7988464736917"))
# # attn_decoder1.load_state_dict(torch.load("0/2TEACHpercep_attn_decoder1__0.18038101494312286_715.1640927611335_0.1423390507698059_699.7988464736917"))
# # encoder1.load_state_dict(torch.load("0/NOTEACHpercep_encoder1__-0.7585325241088867_955.6725159312418"))
# # attn_decoder1.load_state_dict(torch.load("0/NOTEACHpercep_attn_decoder1__-0.7585325241088867_955.6725159312418"))
# trainIters(pairs, v_pairs, T_pairs, encoder1, attn_decoder1, 20*len(pairs), print_every=len(pairs), learning_rate=lr, num = 3)
# outfile = open('%d/onlyteach-perce-model-out-test.fr-en.en' % 0, 'w')
# sys.stdout = outfile
# evaluateRandomly(encoder1, attn_decoder1,100)
# outfile.close()

encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN_log(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
# # encoder1.load_state_dict(torch.load("data/model/pro_encoder1_0_13.57414722442627_746.3501116239448_13.328173637390137_728.678738916424"))
# # attn_decoder1.load_state_dict(torch.load('data/model/pro_attn_decoder1_0_13.57414722442627_746.3501116239448_13.328173637390137_728.678738916424'))
# # encoder1.load_state_dict(torch.load("data/model/pro_encoder1"))
# # attn_decoder1.load_state_dict(torch.load('data/model/pro_attn_decoder1'))
# encoder1.load_state_dict(torch.load("0/NOTECHpro_encoder1__7.145392894744873_573.9545035095189_6.593526840209961_543.7455929463542"))
# attn_decoder1.load_state_dict(torch.load("0/NOTECpro_attn_decoder1__7.145392894744873_573.9545035095189_6.593526840209961_543.7455929463542"))
# encoder1.load_state_dict(torch.load("0/pro_encoder1__14.516222953796387_686.4530524683156_12.21976089477539_663.7829194921254"))
# attn_decoder1.load_state_dict(torch.load('0/pro_attn_decoder1__14.516222953796387_686.4530524683156_12.21976089477539_663.7829194921254'))
# encoder1.load_state_dict(torch.load("0/ONlYTEACH-pro_encoder1__22.244829177856445_652.5906224422009_20.173337936401367_658.9654386139466"))
# attn_decoder1.load_state_dict(torch.load("0/ONLYTEACH-pro_attn_decoder1__22.244829177856445_652.5906224422009_20.173337936401367_658.9654386139466"))
# encoder1.load_state_dict(torch.load("0/NOTEACH-pro_encoder1__14.986833572387695_689.4869410266194_12.674849510192871_674.7815071154189"))
# attn_decoder1.load_state_dict(torch.load("0/NOTEACH-pro_attn_decoder1__14.986833572387695_689.4869410266194_12.674849510192871_674.7815071154189"))
# encoder1.load_state_dict(torch.load("14/ONLYTEACH-pro_encoder1__24.18207550048828_648.737279042414_21.028303146362305_665.1957092220754_25.0"))
# attn_decoder1.load_state_dict(torch.load("14/ONLYTEACH-pro_attn_decoder1__24.18207550048828_648.737279042414_21.028303146362305_665.1957092220754_25.0"))
trainIters_log(pairs, v_pairs, T_pairs,encoder1, attn_decoder1, 100*len(pairs), print_every=len(pairs), learning_rate=0.01, num = 19)
