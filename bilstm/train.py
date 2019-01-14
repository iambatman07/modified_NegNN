# -*-coding:utf-8-*-
#! /usr/bin/env python

from bilstm import BiLSTM
# from random import shuffle
from NegNN.utils.tools import shuffle, padding
from NegNN.utils.metrics import *
from NegNN.processors import int_processor
from NegNN.processors import ext_processor


import tensorflow as tf
import sys
import os
import time
import codecs
import numpy as np

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 50)")
tf.flags.DEFINE_integer("max_sent_length", 100, "Maximum sentence length for padding (default:100)")
tf.flags.DEFINE_integer("num_hidden", 200, "Number of hidden units per layer (default:200)")
tf.flags.DEFINE_integer("num_classes", 2, "Number of y classes (default:2)")
# Training parameters
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 50)")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate(default: 1e-4)")
tf.flags.DEFINE_boolean("scope_detection", True, "True if the task is scope detection or joined scope/event detection")
tf.flags.DEFINE_boolean("event_detection", False, "True is the task is event detection or joined scope/event detection")
tf.flags.DEFINE_integer("POS_emb",0,"0: no POS embeddings; 1: normal POS; 2: universal POS")
tf.flags.DEFINE_boolean("emb_update",False,"True if input embeddings should be updated (default: False)")
tf.flags.DEFINE_boolean("normalize_emb",False,"True to apply L2 regularization on input embeddings (default: False)")
# Data Parameters
tf.flags.DEFINE_string("test_set",'', "Path to the test filename (to use only in test mode")
tf.flags.DEFINE_boolean("pre_training", False, "True to use pretrained embeddings")
tf.flags.DEFINE_string("training_lang",'en', "Language of the tranining data (default: en)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def store_config(_dir,flags):
    with codecs.open(os.path.join(_dir,'config.ini'),'wb','utf8') as _config:
        for attr, value in sorted(FLAGS.__flags.items()):
            _config.write("{}={}\n".format(attr.upper(), value))

# Timestamp and output dir for current model
fold_name = "%s%s_%semb%dnh%d" % ('PRE' if FLAGS.pre_training else "noPRE",
'upd' if FLAGS.pre_training and FLAGS.emb_update else '',
'uniPOS' if FLAGS.POS_emb==2 else 'wONLY',
FLAGS.embedding_dim,
FLAGS.num_hidden)

out_dir = os.path.abspath(os.path.join(os.path.curdir, "NegNN","runs", fold_name))
print "Writing to {}\n".format(out_dir)

# Set checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
store_config(checkpoint_dir,FLAGS)

# Data Preparation
# ==================================================

# Load data
if not FLAGS.pre_training:
    train_set, valid_set, voc, dic_inv = int_processor.load_train_dev(FLAGS.scope_detection, FLAGS.event_detection, FLAGS.training_lang, checkpoint_dir)
    vocsize = len(voc['w2idxs'])
    tag_voc_size = len(voc['t2idxs']) if FLAGS.POS_emb == 1 else len(voc['tuni2idxs'])
else:
    train_set, valid_set, dic_inv, pre_emb_w, pre_emb_t = ext_processor.load_train_dev(FLAGS.scope_detection, FLAGS.event_detection, FLAGS.training_lang, FLAGS.embedding_dim, FLAGS.POS_emb)
    vocsize = pre_emb_w.shape[0]
    tag_voc_size = pre_emb_t.shape[0]

train_lex, train_tags, train_tags_uni, train_cue, _, train_y = train_set
valid_lex, valid_tags, valid_tags_uni, valid_cue, _, valid_y = valid_set

# Training
# ==================================================

def feeder(_bilstm, lex, cue, tags, _y, train = True):
    X = padding(lex, FLAGS.max_sent_length, vocsize - 1)
    C = padding(cue, FLAGS.max_sent_length, 2)
    if tags != []: 
        T = padding(tags, FLAGS.max_sent_length, tag_voc_size - 1)
    Y = padding(numpy.asarray(map(lambda x: [1,0] if x == 0 else [0,1],_y)).astype('int32'),FLAGS.max_sent_length,0,False)
    _mask = [1 if t!=vocsize - 1 else 0 for t in X]
    feed_dict={
        _bilstm.x: X,
        _bilstm.c: C,
        _bilstm.y: Y,
        _bilstm.istate_fw: numpy.zeros((1, 2*FLAGS.num_hidden)),
        _bilstm.istate_bw: numpy.zeros((1, 2*FLAGS.num_hidden)),
        _bilstm.seq_len: numpy.asarray([len(lex)]),
        _bilstm.mask: _mask}
    if tags != []:
        feed_dict.update({_bilstm.t:T})
    if train:
    	feed_dict.update({_bilstm.lr:clr})
    	_, loss_train = sess.run([optimizer, bi_lstm.loss], feed_dict = feed_dict)
    	return loss_train
    else:
    	acc_test, pred = sess.run([bi_lstm.accuracy,bi_lstm.label_out], feed_dict = feed_dict)
    	return acc_test, pred , Y

clr = FLAGS.learning_rate

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        bi_lstm = BiLSTM(num_hidden=FLAGS.num_hidden,
                num_classes=FLAGS.num_classes,
                voc_dim=vocsize,
                emb_dim=FLAGS.embedding_dim,
                sent_max_len = FLAGS.max_sent_length,
                tag_voc_dim = tag_voc_size,
                tags = True if FLAGS.POS_emb in [1,2] else False,
                external = FLAGS.pre_training,
                update = FLAGS.emb_update)

        # Define Training procedure
        optimizer = tf.train.AdamOptimizer(clr).minimize(bi_lstm.loss)

        saver = tf.train.Saver()

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        if FLAGS.pre_training:
            sess.run(bi_lstm._weights['w_emb'].assign(pre_emb_w))
            if FLAGS.POS_emb in [1,2]:
                sess.run(bi_lstm._weights['t_emb'].assign(pre_emb_t))

        train_tot_loss = []
        dev_tot_acc = []
        best_f1 = 0.0
        for e in xrange(FLAGS.num_epochs):

            # shuffle
            if FLAGS.POS_emb in [1,2]: shuffle([train_lex, train_tags, train_tags_uni, train_cue, train_y], 20)
            else: shuffle([train_lex,train_cue,train_y], 20)

            # TRAINING STEP
            train_step_loss = []
            dev_tot_acc = []
            tic = time.time()
            for i in xrange(len(train_lex)):
                if FLAGS.POS_emb in [1,2]:
                    loss_train = feeder(bi_lstm, train_lex[i],train_cue[i], train_tags[i] if FLAGS.POS_emb == 1 else train_tags_uni[i], train_y[i])
                else:
                    loss_train = feeder(bi_lstm, train_lex[i], train_cue[i], [], train_y[i])
                # Calculating batch accuracy
                train_step_loss.append(loss_train)
                print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./len(train_lex)),'completed in %.2f (sec) <<\r'%(time.time()-tic),
                sys.stdout.flush()
            train_tot_loss.append(sum(train_step_loss)/len(train_step_loss))
            print "TRAINING MEAN LOSS: ", train_tot_loss[e]

            # DEVELOPMENT STEP
            pred_dev = []
            gold_dev = []
            for i in xrange(len(valid_lex)):
                if FLAGS.POS_emb in [1,2]:
                    acc_dev, pred, Y_dev = feeder(bi_lstm, valid_lex[i],valid_cue[i],valid_tags[i] if FLAGS.POS_emb == 1 else valid_tags_uni[i],valid_y[i],train=False)
                else:
                    acc_dev, pred, Y_dev = feeder(bi_lstm, valid_lex[i],valid_cue[i], [], valid_y[i],train=False)
                pred_dev.append(pred[:len(valid_lex[i])])
                gold_dev.append(Y_dev[:len(valid_lex[i])])
            f1,rep_dev,cm_dev,f1_pos = get_eval(pred_dev,gold_dev)
            dev_tot_acc.append(f1_pos)

            # STORE TRAINING LOSS AND DEV ACCURACIES
            numpy.save(os.path.join(checkpoint_dir,'train_loss.npy'),train_tot_loss)
            numpy.save(os.path.join(checkpoint_dir,'valid_acc.npy'),dev_tot_acc)

            # STORE INTERMEDIATE RESULTS
            if f1 > best_f1:
                best_f1 = f1
                print "Best f1 is: ",best_f1
                be = e
                # store the model
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                saver.save(sess, checkpoint_prefix,global_step=be)
                print "Model saved."
                write_report(checkpoint_dir, rep_dev, cm_dev, 'dev')
                store_prediction(checkpoint_dir, valid_lex, dic_inv, pred_dev, gold_dev, 'dev')
                dry = 0
            else:
                dry += 1

            if abs(be-e) >= 10 and dry>=5:
                print "Halving the lr..."
                clr *= 0.5
                dry = 0
