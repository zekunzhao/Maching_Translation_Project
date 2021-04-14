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
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def readOutputdata(lang1, lang2, model, reverse=False, dataset = 'test'):
    print("Reading lines...")

    if dataset == 'test':

        # Read the file and split into lines
        lines = open('0/out-test.%s-%s.%s' % (lang1, lang2,lang2), encoding='utf-8').\
            read().strip().split('\n')
        lines2 = open('0/%s-out-test.%s-%s.%s' % (model, lang1, lang2,lang2), encoding='utf-8').\
            read().strip().split('\n')

        empty = 0
        ratio = [len(lines2[item])/len(lines[item]) for item in range(len(lines))]
        for item in range(len(lines)):
            if len(lines2[item])==0:
                empty=empty+1


    print(sum(ratio)/len(ratio))
    print("empty sentences:", empty)
    return 
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
    # input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse,dataset)
    input_lang, output_lang, pairs = readTestdata('fr', 'en', True, dataset)
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

# # input_lang, output_lang, pairs = prepareData('fr', 'en', True)
# input_lang, output_lang, pairs = prepareData('eng', 'fra', True,'original')
# # pairs = readTestdata('fr', 'en', True, 'train')
# # v_pairs = readTestdata('fr', 'en', True, 'val')
# # T_pairs = readTestdata('fr', 'en', True, 'test')
# readOutputdata('fr', 'en', model='pro-model', reverse=False, dataset = 'test')
# readOutputdata('fr', 'en', model='noteach-perce-model', reverse=False, dataset = 'test')
# readOutputdata('fr', 'en', model='perce-model', reverse=False, dataset = 'test')

input_lang, output_lang, pairs = prepareData('eng', 'fra', True,'train')
input_lang, output_lang, pairs = prepareData('eng', 'fra', True,'val')
input_lang, output_lang, pairs = prepareData('eng', 'fra', True,'test')