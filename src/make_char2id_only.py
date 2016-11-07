#!/usr/bin/env python

'''
Usage:
python jws_lstm.py config.ini
'''

import sys
import re
import os
import math
import numpy as np
import configparser
import chainer
import pickle
from chainer import functions as F
from chainer import optimizers
from chainer import FunctionSet, Variable, cuda
from collections import defaultdict


def make_vocab():
    vocab = dict()
    for f in [train_file, test_file]:
        for line in open(f):
            sent = list(''.join(line.rstrip().split(' ')))
            for i in range(window//2 + 3-1):
                sent.append('</s>')
                sent.insert(0,'<s>')
            for i in range(3-1, len(sent)-(3-1)+2):
                uni_gram = sent[i]
                bi_gram = sent[i-1] + sent[i]
                tri_gram = sent[i-2] + sent[i-1] + sent[i]
                if uni_gram not in vocab:
                    vocab[uni_gram] = len(vocab)
                if bi_gram not in vocab:
                    vocab[bi_gram] = len(vocab)
                if tri_gram not in vocab:
                   vocab[tri_gram] = len(vocab)
    vocab['UNK'] = len(vocab) 
    return vocab

if __name__ == '__main__':
    # reading config
    ini_file = sys.argv[1]
    ini = configparser.SafeConfigParser()
    ini.read(ini_file)
    train_file = ini.get('Data', 'train')
    test_file = ini.get('Data', 'test')
    dict_file = ini.get('Data', 'dict')
    result_raw = ini.get('Result', 'raw')
    config_file = ini.get('Result', 'config')
    evaluation = ini.get('Result', 'evaluation')
    model_file = ini.get('Result', 'model')
    window = int(ini.get('Parameters', 'window'))
    embed_units = int(ini.get('Parameters', 'embed_units'))
    char_type_embed_units = int(ini.get('Parameters', 'char_type_embed_units'))
    hidden_units = int(ini.get('Parameters', 'hidden_units'))
    dict_embed_units = int(ini.get('Parameters', 'dict_embed_units'))
    lam = float(ini.get('Parameters', 'lam'))
    label_num = int(ini.get('Settings', 'label_num'))
    batch_size = int(ini.get('Settings', 'batch_size'))
    learning_rate = float(ini.get('Parameters', 'learning_rate'))
    dropout_rate = float(ini.get('Parameters', 'dropout_rate'))
    n_epoch = int(ini.get('Settings', 'n_epoch'))
    delta = float(ini.get('Parameters', 'delta'))
    opt_selection = ini.get('Settings', 'opt_selection')
    with open(config_file, 'w') as config:
        ini.write(config)

    char2id = make_vocab()
    with open('char2id.model', 'wb') as char2id_file:
        pickle.dump(char2id, char2id_file)

