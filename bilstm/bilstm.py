#-*- coding: utf-8-*-
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from NegNN.utils.metrics import *
from NegNN.utils.tools import *

import numpy
import random
import codecs
import os,sys
import time
import subprocess

class BiLSTM(object):
    def __init__(self,
            num_hidden,
            num_classes,
            voc_dim,
            emb_dim,
            sent_max_len,
            tag_voc_dim,
            tags,
            external,
            update):

        # tf Graph
        # shared parameters
        self.num_hidden = num_hidden
        self.sent_max_len = sent_max_len
        # input placeholders
        self.seq_len = tf.placeholder(tf.int64,name="input_lr")
        self.lr = tf.placeholder(tf.float32)
        self.x = tf.placeholder(tf.int32,name="input_x")
        self.c = tf.placeholder(tf.int32,name="input_c")
        if tags: self.t = tf.placeholder(tf.int32,name="input_t")
        self.mask = tf.placeholder("float",name="input_mask")

        # Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
        self.istate_fw = tf.placeholder("float", [None, 2*num_hidden])
        self.istate_bw = tf.placeholder("float", [None, 2*num_hidden])
        self.y = tf.placeholder("float", [None, num_classes])

        # Define weights
        self._weights = {
            # Hidden layer weights => 2*n_hidden because of foward + backward cells
            'w_emb' : tf.Variable(0.2 * tf.random_uniform([voc_dim,emb_dim], minval=-1.0, maxval=1.0, dtype=tf.float32),name='w_emb',trainable=update),
            'c_emb' : random_uniform([3,emb_dim],'c_emb')
            }
        if tags:
            self._weights.update({'t_emb' : tf.Variable(0.2 * tf.random_uniform([tag_voc_dim,emb_dim], minval=-1.0, maxval=1.0, dtype=tf.float32),name='t_emb',trainable=update)})
        else:
            self._weights = {
                'w_emb' : random_uniform([voc_dim, emb_dim],'w_emb'),
                'c_emb' : random_uniform([3,emb_dim],'c_emb')
                }
            if tags:
                self._weights.update({'t_emb' : random_uniform([tag_voc_dim,emb_dim],'t_emb')})

        self._weights.update({
            'hidden_w': tf.Variable(tf.random_normal([emb_dim, 2*num_hidden])),
            'hidden_c': tf.Variable(tf.random_normal([emb_dim, 2*num_hidden])),
            'out_w': tf.Variable(tf.random_normal([2*num_hidden, num_classes]))
                })
        if tags:
                self._weights.update({'hidden_t': tf.Variable(tf.random_normal([emb_dim, 2*num_hidden]))})

        self._biases = {
            'hidden_b': tf.Variable(tf.random_normal([2*num_hidden])),
            'out_b': tf.Variable(tf.random_normal([num_classes]))
        }

        # self.normalize_w_emb = tf.nn.l2_normalize(self._weights['w_emb'],1)
        # # if tags: self.normalize_t_emb = tf.nn.l2_normalize(self._weights['t_emb'],1)

        if tags:
            self.pred = self.BiLSTMgraph(self.x, self.c, self.t, self.istate_fw, self.istate_bw, self._weights, self._biases)
        else:
            self.pred = self.BiLSTMgraph(self.x, self.c, None, self.istate_fw, self.istate_bw, self._weights, self._biases)

        pred_mod = [tf.matmul(item, self._weights['out_w']) + self._biases['out_b'] for item in self.pred]
        outputs = tf.squeeze(tf.pack(pred_mod))

        self.loss = tf.reduce_sum(tf.mul(tf.nn.softmax_cross_entropy_with_logits(outputs, self.y),self.mask))/tf.reduce_sum(self.mask) # softmax

        self.label_out = tf.nn.softmax(outputs,name="predictions")

        self.accuracy = tf.equal(tf.argmax(self.label_out,1), tf.argmax(self.y,1),name="accuracy")

    def BiLSTMgraph(self, _X, _C, _T, _istate_fw, _istate_bw, _weights, _biases):
        # input: a [len_sent,len_seq] (e.g. 7x5)
        # transform into embeddings
        if _T:
            emb_x = tf.nn.embedding_lookup(self._weights['w_emb'],_X)
            emb_c = tf.nn.embedding_lookup(self._weights['c_emb'],_C)
            emb_t = tf.nn.embedding_lookup(self._weights['t_emb'],_T)
            # Linear activation
            _X = tf.matmul(emb_x, self._weights['hidden_w']) + tf.matmul(emb_c, self._weights['hidden_c']) + tf.matmul(emb_t,self._weights['hidden_t']) + self._biases['hidden_b']
        else:
            emb_x = tf.nn.embedding_lookup(self._weights['w_emb'],_X)
            emb_c = tf.nn.embedding_lookup(self._weights['c_emb'],_C)
            # Linear activation
            _X = tf.matmul(emb_x, self._weights['hidden_w']) + tf.matmul(emb_c,self._weights['hidden_c']) + self._biases['hidden_b']

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = rnn_cell.BasicLSTMCell(self.num_hidden, forget_bias=1.0)
        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.5)
        # Backward direction cell
        lstm_bw_cell = rnn_cell.BasicLSTMCell(self.num_hidden, forget_bias=1.0)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=0.5)
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(0,self.sent_max_len,_X)

        # Get lstm cell output
        outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, _X,initial_state_fw = self.istate_fw, initial_state_bw=self.istate_bw, sequence_length = self.seq_len)

        return outputs
