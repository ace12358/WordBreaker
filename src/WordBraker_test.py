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
import pickle
import configparser
import chainer
from chainer import functions as F
from chainer import optimizers
from chainer import FunctionSet, Variable, cuda
from chainer import serializers

def make_vocab():
    vocab = dict()
    for f in [train_file, test_file]:
        for line in open(f):
            pre_char = '<s>'
            for char in ''.join(line.strip().split()):
                bi_gram = pre_char + char
                if char not in vocab:
                    vocab[char] = len(vocab)
                if bi_gram not in vocab:
                    vocab[bi_gram] = len(vocab)
                pre_char = char
            bi_gram = char + '</s>'
            if bi_gram not in vocab:
                vocab[bi_gram] = len(vocab)
    vocab['<s>'] = len(vocab)
    vocab['</s>'] = len(vocab)
    vocab['<s><s>'] = len(vocab)
    vocab['</s></s>'] = len(vocab)
    vocab['UNK'] = len(vocab)
    
    return vocab

def make_char_type(char):
    if re.match(u'[ぁ-ん]', char):
        return('hiragana')
    elif re.match(u'[ァ-ン]', char):
        return('katakana')
    elif re.match(u'[一-龥]', char):
        return('kanji')
    elif re.match(u'[A-Za-z]', char):
        return('alphabet')
    elif re.match(u'[0-9０-９]', char):
        return('number')
    elif char=='<s>':
        return char
    elif char=='</s>':
        return char
    else:
        return('other')

def make_char_type2id():
    char_type2id = dict()
    char_type2id['hiragana'] = len(char_type2id)
    char_type2id['katakana'] = len(char_type2id)
    char_type2id['kanji'] = len(char_type2id)
    char_type2id['alphabet'] = len(char_type2id)
    char_type2id['number'] = len(char_type2id)
    char_type2id['other'] = len(char_type2id)
    char_type2id['<s>'] = len(char_type2id)
    char_type2id['</s>'] = len(char_type2id)
    for pre in ['hiragana','katakana','kanji','alphabet','number','other','<s>','</s>']:
        for cur in ['hiragana','katakana','kanji','alphabet','number','other','<s>','</s>']:
            char_type2id[pre+cur] = len(char_type2id)
    return char_type2id


def forward_one(x,target, hidden, prev_c, model):
    # make input window vector
    distance = window // 2
    char_vecs = list()
    char_type_vecs = list()
    x = list(x)
    for i in range(distance):
        x.append('</s>')
        x.append('</s>')
        x.insert(0,'<s>')
        x.insert(0,'<s>')
    for i in range(-distance , distance+1):
        char = x[target+2 + i]
        try:
            char_id = char2id[char]
        except(KeyError):
            char_id = char2id['UNK']
            
        char_vec = model.embed(get_onehot(char_id))
        char_vecs.append(char_vec)
        bi_gram = x[target+2+i] + x[target+2+i+1]
        try:
            bi_gram_id = char2id[bi_gram]
        except(KeyError):
            bi_gram_id = char2id['UNK']
        bi_gram_char_vec = model.embed(get_onehot(bi_gram_id))
        char_vecs.append(bi_gram_char_vec)
    char_concat = F.concat(tuple(char_vecs))
    for i in range(-distance, distance+1):
        char = x[target+2+ i]
        pre_char = x[target+2+ i + 1]
        char_type = make_char_type(char)
        pre_char_type = make_char_type(pre_char)
        bi_gram_type = pre_char_type + char_type
        char_type_id = char_type2id[char_type]
        bigram_type_id = char_type2id[bi_gram_type]
        char_type_vec = model.char_type_embed(get_onehot(char_type_id))
        bigram_type_vec = model.char_type_embed(get_onehot(bigram_type_id))
        char_type_vecs.append(char_type_vec)
        char_type_vecs.append(bigram_type_vec)
    char_type_concat = F.concat(tuple(char_type_vecs))
    #dropout_concat = F.dropout(concat, ratio=dropout_rate, train=train_flag)
    concat = F.concat((char_concat, char_type_concat))
    concat = F.concat((concat, hidden))
    i_gate = F.sigmoid(model.i_gate(concat))
    f_gate = F.sigmoid(model.f_gate(concat))
    o_gate = F.sigmoid(model.o_gate(concat))
    concat = F.concat((hidden, i_gate, f_gate, o_gate))
    prev_c, hidden = F.lstm(prev_c, concat)
    output = model.output(hidden)
    dist = F.softmax(output)
    return np.argmax(dist.data)

def __call__(char2id,model_file):
    with open(model_file, 'rb') as mf:
        model = pickle.load(mf)
    labels = list()
    line_cnt = 0
    for line in sys.stdin:
        line_cnt += 1
        hidden = chainer.Variable(np.zeros((1, hidden_units),\
                                                dtype=np.float32))
        prev_c = chainer.Variable(np.zeros((1, hidden_units),\
                                                dtype=np.float32))
        x = ''.join(line.strip().split())
        dists = list()
        for target in range(len(x)):
            dist = forward_one(x, target, hidden, prev_c, model)
            dists.append(dist)
        print("{0}".format(''.join(label2seq(x, dists))))
        labels = list()
   
def get_onehot(num):
    return chainer.Variable(np.array([num], dtype=np.int32))

def label2seq(x, labels):
    seq = list()
    for i in range(len(x)):
        if label_num == 2:
            if i == 0:
                seq.append(x[i])
            elif labels[i] == 0:
                seq.append(' ')
                seq.append(x[i])
            else:
                seq.append(x[i])
        elif label_num == 3:
            if i == 0:
                seq.append(x[i])
            elif labels[i] == 0 or labels[i] == 2:
                seq.append(' ')
                seq.append(x[i])
            else:
                seq.append(x[i])
        elif label_num == 4:
            if i == 0:
                seq.append(x[i])
            elif labels[i] == 0 or labels[i] == 3:
                seq.append(' ')
                seq.append(x[i])
            else:
                seq.append(x[i])
    return seq

if __name__ == '__main__':
    # reading config
    ini_file = sys.argv[1]
    ini = configparser.SafeConfigParser()
    ini.read(ini_file)
    train_file = ini.get('Data', 'train')
    test_file = ini.get('Data', 'test')
    result_raw = ini.get('Result', 'raw')
    config_file = ini.get('Result', 'config')
    evaluation = ini.get('Result', 'evaluation')
    model = ini.get('Result', 'model')
    window = int(ini.get('Parameters', 'window'))
    embed_units = int(ini.get('Parameters', 'embed_units'))
    char_type_embed_units = int(ini.get('Parameters', 'char_type_embed_units'))
    hidden_units = int(ini.get('Parameters', 'hidden_units'))
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
    char_type2id = make_char_type2id()
    #model, opt = init_model(len(char2id), len(char_type2id))
    #test(char2id, mode
    __call__(char2id, model)
