# -*-coding:utf-8-*-
#! /usr/bin/env python

import feed_forw,feed_forw_tags
import tensorflow as tf
import sys,os
import time

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
tf.flags.DEFINE_boolean("pre_training", False, "True to use pretrained embeddings")
tf.flags.DEFINE_string("training_lang",'en', "Language of the tranining data (default: en)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def store_config(_dir,flags):
    with open(os.path.join(_dir,'config.ini'),'wb') as _config:
        for attr, value in sorted(FLAGS.__flags.items()):
            _config.write("{}={}\n".format(attr.upper(), value))

# Timestamp and output dir for current model
fold_name = "%s%s_%d%s" % ('PRE' if FLAGS.pre_training else "noPRE",
'upd' if FLAGS.pre_training and FLAGS.emb_update else '',
FLAGS.POS_emb,str(int(time.time())))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "NegNN","runs", fold_name))
print "Writing to {}\n".format(out_dir)

# Set checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
store_config(checkpoint_dir,FLAGS)

if FLAGS.POS_emb == 0:
    feed_forw.ff(scope_dect = FLAGS.scope_detection,
            event_dect = FLAGS.event_detection,
            tr_lang = FLAGS.training_lang,
            folder = checkpoint_dir,
            clr = FLAGS.learning_rate,
            n_hidden = FLAGS.num_hidden,
            n_classes = FLAGS.num_classes,
            nepochs = FLAGS.num_epochs,
            emb_size = FLAGS.embedding_dim,
            POS_emb = FLAGS.POS_emb,
            max_sent_len = FLAGS.max_sent_length,
            update = FLAGS.emb_update,
            pre_training = FLAGS.pre_training,
            training = True,
            test_files = None,
            test_lang = None)
else:
    feed_forw_tags.ff_tags(scope_dect = FLAGS.scope_detection,
            event_dect = FLAGS.event_detection,
            tr_lang = FLAGS.training_lang,
            folder = checkpoint_dir,
            clr = FLAGS.learning_rate,
            n_hidden = FLAGS.num_hidden,
            n_classes = FLAGS.num_classes,
            nepochs = FLAGS.num_epochs,
            emb_size = FLAGS.embedding_dim,
            POS_emb = FLAGS.POS_emb,
            max_sent_len = FLAGS.max_sent_length,
            update = FLAGS.emb_update,
            pre_training = FLAGS.pre_training,
            training = True,
            test_files = None,
            test_lang = None)