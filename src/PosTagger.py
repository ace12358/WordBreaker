#!/usr/bin/env python

'''
Usage:
python jws_lstm.py config.ini
'''
import time
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
    word_pos = dict()
    for f in [train_file, test_file]:
        for line in open(f):
            words = list()
            sent_wp = line.rstrip().split(' ')
            for wordpos in sent_wp:
                word_pos[wordpos] = 1
                word = wordpos.split('/')[0]
                words.append(word)
                if 'word:'+word not in vocab:
                    vocab['word:'+word]=len(vocab)
            sent = words
            sent = list(''.join(words))
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
            for j in range(len(words)):
                sent = words[:j] + words[j+1:]
                sent = list(''.join(sent))
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
    return vocab, word_pos



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
        #dict_embed_units),
        hidden1=F.Linear(window * (embed_units + char_type_embed_units)*3 +embed_units + hidden_units, hidden_units),
        i_gate=F.Linear(window * (embed_units + char_type_embed_units)*3 +embed_units +hidden_units, hidden_units),
        f_gate=F.Linear(window * (embed_units + char_type_embed_units)*3 + embed_units+hidden_units, hidden_units),
        o_gate=F.Linear(window * (embed_units + char_type_embed_units)*3 + embed_units+hidden_units, hidden_units),
        output=F.Linear(hidden_units + len(char_type2id)+ 23 , 23),
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


def pos2id():
    pos2id = dict()
    for pos in open(pos_file):
        pos2id[pos.rstrip()] = len(pos2id)
    return pos2id

def id2pos(pos2id):
    id2pos = dict()
    for pos, id in pos2id.items():
        id2pos[id] = pos
    return id2pos



def forward_one(x, target,model,  label, hidden, prev_c, train_flag):

    # make input window vector
    distance =  window // 2
    s_num = 3-1 + window // 2
    char_vecs = list()
    char_type_vecs = list()
    words = list()
    for wp in x:
        w = wp.split('/')[0]
        words.append(w)

    c_index_r = 0
    for i in range(target+1):
        c_index_r += len(words[i])
    c_index_l = c_index_r - len(words[target])

    x = words
    target_word = words[i]
    c = list(''.join(words))

    x = c[:c_index_l] + c[c_index_r:]
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
        uni_gram_id = char2id[uni_gram]
        bi_gram_id = char2id[bi_gram]
        tri_gram_id = char2id[tri_gram]
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
    
    # word feature
    target_word_id = char2id['word:'+target_word]
    word_vec = model.embed(get_onehot(target_word_id))
    ct_word = list()
    for c in target_word:
        ct_word.append(make_char_type(c))
    try:
        ct_index = char_type2id[''.join(list(set(ct_word)))]
    except:
        ct_index = char_type2id['other']
    one_hot = list()
    one_hot = [0]*len(char_type2id)
    one_hot[ct_index] = 1
    #for i in range(len(char_type2id)):
    #    if i == ct_index:
    #        one_hot.append(1)
    #    else:
    #        one_hot.append(0)
    
    pos_d = list()
    for i in range(23):
        if target_word+'/'+id2pos[i] in word_pos:
            pos_d.append(1)
        else:
            pos_d.append(0)
    #pos_d = [0]*23
    
    ct_word_vec = chainer.Variable(np.array([one_hot], dtype=np.float32))
    pos_d_vec = chainer.Variable(np.array([pos_d], dtype=np.float32))
    
    char_concat = F.concat(tuple(char_vecs))
    char_type_concat = F.concat(tuple(char_type_vecs))
    #dropout_concat = F.dropout(concat, ratio=dropout_rate, train=train_flag)
    concat = F.concat((char_concat, char_type_concat, word_vec))
    concat = F.concat((concat, hidden))
    i_gate = F.sigmoid(model.i_gate(concat))
    f_gate = F.sigmoid(model.f_gate(concat))
    o_gate = F.sigmoid(model.o_gate(concat))
    concat = F.concat((hidden, i_gate, f_gate, o_gate))
    prev_c, hidden = F.lstm(prev_c, concat)
    output = model.output(F.concat((hidden, ct_word_vec, pos_d_vec)))
    dist = F.softmax(output)
    #print(dist.data, label, np.argmax(dist.data))
    #print(output.dataect.data)
    return np.argmax(dist.data)

        
def main(char2id, model_file):
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
        x = line.rstrip().split(' ')
        dists = list()
        for target in range(len(x)):
            pos = x[target].split('/')[0]
            dist = forward_one(x, target, model, pos, hidden, prev_c , train_flag=True)
            dists.append(dist)
        print("{0}".format((merge_wp(x, dists))))

def get_onehot(num):
    return chainer.Variable(np.array([num], dtype=np.int32))


def merge_wp(x, poss):
    result = list()
    for i in range(len(x)):
        wp = x[i].split('/')[0]+'/'+id2pos[poss[i]]
        result.append(wp)
    return ' '.join(result)

if __name__ == '__main__':
    # reading config
    ini_file = sys.argv[1]
    ini = configparser.SafeConfigParser()
    ini.read(ini_file)
    train_file = ini.get('Data', 'train')
    test_file = ini.get('Data', 'test')
    pos_file = ini.get('Data', 'pos_file')
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
    batch_size = int(ini.get('Settings', 'batch_size'))
    learning_rate = float(ini.get('Parameters', 'learning_rate'))
    dropout_rate = float(ini.get('Parameters', 'dropout_rate'))
    n_epoch = int(ini.get('Settings', 'n_epoch'))
    delta = float(ini.get('Parameters', 'delta'))
    opt_selection = ini.get('Settings', 'opt_selection')
    pos2id = pos2id()
    id2pos = id2pos(pos2id)
    with open(config_file, 'w') as config:
        ini.write(config)

    char2id, word_pos = make_vocab()
    char_type2id = make_char_type2id()
    main(char2id, model_file)
    #test(char2id, model)

