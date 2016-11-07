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
    with open(char2id_file, 'rb') as char2id:
        vocab = pickle.load(char2id)
    return vocab

def make_word_dict():
    words = list()
    word2freq = defaultdict(lambda:0)
    for f in [train_file, test_file]:
    #for f in [train_file]:
        for line in open(f):
            ws = (line.rstrip().split(' '))
            for w in ws:
                word2freq[w] += 1
    for word, freq in word2freq.items():
        if freq > 1:
            words.append(word)
    for f in [dict_file]:
        for w in f:
            words.append(w.rstrip())
    words = set(words)
    return words

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
    for one in ['hiragana','katakana','kanji','alphabet','number','other','<s>','</s>']:
        for two in ['hiragana','katakana','kanji','alphabet','number','other','<s>','</s>']:
            char_type2id[one+two] = len(char_type2id)
            for three in ['hiragana','katakana','kanji','alphabet','number','other','<s>','</s>']:
                char_type2id[one+two+three] = len(char_type2id)
    return char_type2id

def init_model(vocab_size, char_type_size):
    model = FunctionSet(
        embed=F.EmbedID(vocab_size, embed_units),
        char_type_embed = F.EmbedID(char_type_size, char_type_embed_units),
        #dict_embed = F.Linear(12, dict_embed_units),
        hidden1=F.Linear(window * (embed_units + char_type_embed_units)*3 + hidden_units, hidden_units),
        i_gate=F.Linear(window * (embed_units + char_type_embed_units)*3 + hidden_units, hidden_units),
        f_gate=F.Linear(window * (embed_units + char_type_embed_units)*3 + hidden_units, hidden_units),
        o_gate=F.Linear(window * (embed_units + char_type_embed_units)*3 + hidden_units, hidden_units),
        output=F.Linear(hidden_units + 12, label_num),
    )
    if opt_selection == 'Adagrad':
        opt = optimizers.AdaGrad(lr=learning_rate)
    elif opt_selection == 'SGD':
        opt = optimizers.SGD()
    elif opt_selection == 'Adam':
        opt = optimizers.Adam()
    else:
        opt = optimizers.AdaGrad(lr=learning_rate)
        print('Adagrad is chosen as defaut')
    opt.setup(model)
    return model, opt


def forward_one(x, target, model, hidden, prev_c, word_dict):
    # make dict feature vector
    dict_vec = list()
    L1 = L2 = L3 = L4 = R1 = R2 = R3 = R4 = I1 = I2 = I3 = I4 = 0
    for i in range(len(x[:target])):
        word_candidate = x[target-(i+1):target]
        if word_candidate in word_dict:
            if len(word_candidate) == 1:
                L1 = 1
            elif len(word_candidate) == 2:
                L2 = 1
            elif len(word_candidate) == 3:
                L3 = 1
            else:
                L4 = 1
        if i == 10:
            break

    for i in range(len(x[target:])):
        word_candidate = x[target:target+i+1]
        if word_candidate in word_dict:
            if len(word_candidate) == 1:
                R1 = 1
            elif len(word_candidate) == 2:
                R2 = 1
            elif len(word_candidate) == 3:
                R3 = 1
            else:
                R4 = 1
        if i == 10:
            break
    
    for i in range(1, 6, 1):
        for j in range(1, 6, 1):
            word_candidate = x[target-i:target+j]
            if word_candidate in word_dict:
                if len(word_candidate) == 1:
                    I1 = 1
                elif len(word_candidate) == 2:
                    I2 = 1
                elif len(word_candidate) == 3:
                    I3 = 1
                else:
                    I4 = 1
    dict_vec = chainer.Variable(np.array([[L1,L2,L3,L4,R1,R2,R3,R4,I1,I2,I3,I4]], dtype=np.float32))
    # dict_embed_vec = model.dict_embed(dict_vec)
    # make input window vector
    distance =  window // 2
    s_num = 3-1 + window // 2
    char_vecs = list()
    char_type_vecs = list()
    x = list(x)
    for i in range(s_num):
        x.append('</s>')
        x.insert(0,'<s>')
    for i in range(-distance, distance+1):

    # make char vector 
        # import char
        uni_gram = x[target+s_num+i]
        bi_gram = x[target+s_num-1+i] + x[target+s_num+i]
        tri_gram = x[target+s_num-2+i] + x[target+s_num-1+i] + x[target+s_num+i]
        # char2id
        try:
            uni_gram_id = char2id[uni_gram]
        except(KeyError):
            uni_gram_id = char2id['UNK']
        try:
            bi_gram_id = char2id[bi_gram]
        except(KeyError):
            bi_gram_id = char2id['UNK']
        try:
            tri_gram_id = char2id[tri_gram]
        except(KeyError):
            tri_gram_id = char2id['UNK']
        # id 2 embedding
        uni_gram_vec = model.embed(get_onehot(uni_gram_id))
        bi_gram_vec = model.embed(get_onehot(bi_gram_id))
        tri_gram_vec = model.embed(get_onehot(tri_gram_id))
        # add all char_vec
        char_vecs.append(uni_gram_vec)
        char_vecs.append(bi_gram_vec)
        char_vecs.append(tri_gram_vec)
    # make char type vector 
        # import char type
        uni_gram_type = make_char_type(uni_gram)
        bi_gram_type = make_char_type(x[target+s_num-1+i]) + make_char_type(x[target+s_num+i])
        tri_gram_type = make_char_type(x[target+s_num-2+i]) + make_char_type(x[target+s_num+i] + make_char_type(x[target+s_num-2+i]))
        # chartype 2 id
        uni_gram_type_id = char_type2id[uni_gram_type]
        bi_gram_type_id =  char_type2id[bi_gram_type]
        tri_gram_type_id = char_type2id[tri_gram_type]
        # id 2 embedding
        uni_gram_type_vec = model.char_type_embed(get_onehot(uni_gram_type_id))
        bi_gram_type_vec = model.char_type_embed(get_onehot(bi_gram_type_id))
        tri_gram_type_vec = model.char_type_embed(get_onehot(tri_gram_type_id))
        # add all char_type_vec
        char_type_vecs.append(uni_gram_type_vec)
        char_type_vecs.append(bi_gram_type_vec)
        char_type_vecs.append(tri_gram_type_vec)

    char_concat = F.concat(tuple(char_vecs))
    char_type_concat = F.concat(tuple(char_type_vecs))
    #dropout_concat = F.dropout(concat, ratio=dropout_rate, train=train_flag)
    concat = F.concat((char_concat, char_type_concat))
    concat = F.concat((concat, hidden))
    i_gate = F.sigmoid(model.i_gate(concat))
    f_gate = F.sigmoid(model.f_gate(concat))
    o_gate = F.sigmoid(model.o_gate(concat))
    concat = F.concat((hidden, i_gate, f_gate, o_gate))
    prev_c, hidden = F.lstm(prev_c, concat)
    output = model.output(F.concat((hidden, dict_vec)))
    dist = F.softmax(output)
    return np.argmax(dist.data)

def main(char2id, model):
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
            dist = forward_one(x, target, model, hidden, prev_c ,word_dict)
            dists.append(dist)
        print("{0}".format(''.join(label2seq(x, dists))))


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
    dict_file = ini.get('Data', 'dict')
    result_raw = ini.get('Result', 'raw')
    config_file = ini.get('Result', 'config')
    evaluation = ini.get('Result', 'evaluation')
    model_file = ini.get('Result', 'model')
    char2id_file = ini.get('Result', 'char2id')
    odel_file = ini.get('Result', 'model') 
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
    word_dict = make_word_dict()
    char_type2id = make_char_type2id()
    main(char2id, model_file)
    #model, opt = init_model(len(char2id), len(char_type2id))
    #test(char2id, model)

