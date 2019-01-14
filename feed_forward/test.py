# -*-coding:utf-8-*-
#! /usr/bin/env python

from NegNN.utils.metrics import *
import feed_forw,feed_forw_tags
import tensorflow as tf
import codecs
import sys,os

# Parameters
# ==================================================
# Model Parameters
tf.flags.DEFINE_string("test_set",'', "Path to the test filename (to use only in test mode")
tf.flags.DEFINE_string("checkpoint_dir",'',"Path to the directory where the last checkpoint is stored")
tf.flags.DEFINE_string("test_lang",'en', "en for English, it for Italian (default: en)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

def retrieve_flags(config_file):
    _config = codecs.open(config_file,'rb','utf8').readlines()
    return dict([(k.lower(),v) for k,v in map(lambda x: x.strip().split('='),_config)])

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
config_file = os.path.join(FLAGS.checkpoint_dir,'config.ini')
model_flags = retrieve_flags(config_file)

def str2bool(x):
    return True if x == "True" else False

scope_detection = str2bool(model_flags['scope_detection'])
event_detection = str2bool(model_flags['event_detection'])
embedding_dim = int(model_flags['embedding_dim'])
POS_emb = int(model_flags['pos_emb'])
pre_training = str2bool(model_flags['pre_training'])
num_classes = int(model_flags['num_classes'])
max_sent_length = int(model_flags['max_sent_length'])
num_hidden = int(model_flags['num_hidden'])
emb_update = str2bool(model_flags['emb_update'])
training_lang = model_flags['training_lang']
learning_rate = model_flags['learning_rate']
nepochs = model_flags['num_epochs']

test_files = FLAGS.test_set.split(',')

if POS_emb == 0:
    feed_forw.ff(scope_dect = scope_detection,
            event_dect = event_detection,
            tr_lang = training_lang,
            clr = learning_rate,
            folder = FLAGS.checkpoint_dir,
            n_hidden = num_hidden,
            n_classes = num_classes,
            nepochs = nepochs,
            emb_size = embedding_dim,
            POS_emb = POS_emb,
            max_sent_len = max_sent_length,
            update = emb_update,
            pre_training = pre_training,
            training = False,
            test_files = test_files,
            test_lang = FLAGS.test_lang)

else:
    feed_forw_tags.ff_tags(scope_dect = scope_detection,
            event_dect = event_detection,
            tr_lang = training_lang,
            clr = learning_rate,
            folder = FLAGS.checkpoint_dir,
            n_hidden = num_hidden,
            n_classes = num_classes,
            nepochs = nepochs,
            emb_size = embedding_dim,
            POS_emb = POS_emb,
            max_sent_len = max_sent_length,
            update = emb_update,
            pre_training = pre_training,
            training = False,
            test_files = test_files,
            test_lang = FLAGS.test_lang)