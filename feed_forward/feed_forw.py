# -*-coding:utf-8-*-

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from NegNN.utils.tools import shuffle, contextwin_lr, random_uniform, unpickle_data
from NegNN.utils.metrics import *
import numpy
import int_processor, ext_processor
import random
import codecs
import os,sys
import time
import subprocess

def ff(scope_dect,
    event_dect,
    tr_lang,
    folder,
    clr,
    n_hidden,
    n_classes,
    emb_size,
    nepochs,
    POS_emb,
    max_sent_len,
    pre_training,
    update,
    training,
    test_files,
    test_lang):

    if training:
        # load data
        if not pre_training:
            train_set, valid_set, dic, dic_inv = int_processor.load_train_dev(scope_dect, event_dect, tr_lang, folder)
            vocsize = len(dic['w2idxs'])
        else:
            train_set, valid_set, dic_inv, pre_emb_w, _ = ext_processor.load_train_dev(scope_dect, event_dect, tr_lang, emb_size, POS_emb)
            vocsize = pre_emb_w.shape[0]

        train_lex, _, _, train_cue, train_scope, train_y = train_set
        valid_lex, _, _, valid_cue, valid_scope, valid_y = valid_set

    else:
        # Load data
        if not pre_training:
            assert test_lang == tr_lang
            _, _, voc, dic_inv = unpickle_data(folder)
            test_lex, _, _, test_cue, _, test_y = int_processor.load_test(test_files, voc, scope_dect, event_dect, test_lang)
        else:
            test_set, dic_inv, pre_emb_w, _ = ext_processor.load_test(test_files, scope_dect, event_dect, test_lang, emb_size, POS_emb)
            test_lex, _, _, test_cue, _, test_y = test_set

        if pre_training: vocsize = pre_emb_w.shape[0]
        else: vocsize = len(voc['w2idxs'])

# **************************************************************************

    WIN_LEFT = 9 # number of words in the context window to the left
    WIN_RIGHT = 16 # number of words in the context window to the right
    WIN_LEN = WIN_LEFT + WIN_RIGHT + 1

    word_emb = random_uniform([vocsize,emb_size],'word_emb')
    cue_emb = random_uniform([2,emb_size],'cue_emb')
    # WIN_LEFT + WIN_RIGHT + 1 plus 1 because it has to include the target word
    Wx = random_uniform([emb_size * WIN_LEN, n_hidden],'Wx')
    Wc = random_uniform([emb_size * WIN_LEN, n_hidden],"Wc")
    W = random_uniform([n_hidden, n_classes],"W")
    b = tf.Variable(tf.zeros([n_hidden]),'b')
    bo = tf.Variable(tf.zeros([n_classes]),'bo')

    # setting the variables
    lr = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.int32, shape=[None, WIN_LEN], name='input_x')
    c = tf.placeholder(tf.int32, shape=[None, WIN_LEN], name='input_c')
    y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name='input_y')


    rsh_x = tf.reshape(x,[-1])
    rsh_c = tf.reshape(c,[-1])
    with tf.device("/cpu:0"):
        emb_x = tf.nn.embedding_lookup(word_emb,rsh_x)
        emb_c = tf.nn.embedding_lookup(cue_emb,rsh_c)

    emb_x_rsh = tf.reshape(emb_x,[-1,emb_size * WIN_LEN])
    emb_c_rsh = tf.reshape(emb_c,[-1,emb_size * WIN_LEN])

    y = tf.nn.softmax(tf.matmul(tf.sigmoid(tf.matmul(emb_x_rsh,Wx) + tf.matmul(emb_c_rsh,Wc) + b), W) + bo)

    cost = -tf.reduce_sum(y_*tf.log(y))
    # train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    # normalize_w_emb = tf.nn.l2_normalize(word_emb,1)
    # normalize_c_emb = tf.nn.l2_normalize(cue_emb,1)

    # for testing purposes
    prediction = tf.argmax(y,1)
    g = y_
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

# **************************************************************

    def feeder(lex, cue, _y, train = True):
        X = contextwin_lr(lex, WIN_LEFT, WIN_RIGHT,'word',vocsize - 1)
        C = contextwin_lr(cue,WIN_LEFT, WIN_RIGHT,'cue')
        Y = numpy.asarray(map(lambda x: [1,0] if x == 0 else [0,1],_y)).astype('int32')
        feed_dict={
            x: X,
            c: C,
            y_: Y}
        if train:
            feed_dict.update({lr:clr})
            _, acc_train = sess.run([optimizer, accuracy], feed_dict = feed_dict)
            return acc_train
        else:
            acc_test, pred = sess.run([accuracy,prediction], feed_dict = feed_dict)
            pred = numpy.asarray(map(lambda x: [1,0] if x == 0 else [0,1], pred)).astype('int32')
	    return acc_test, pred , Y
            # acc_test = sess.run(accuracy, feed_dict = feed_dict)
            # return acc_test, Y

    saver = tf.train.Saver(tf.all_variables())
    # Launch the session  
    with tf.Session() as sess:
        if training:
            # saver = tf.train.Saver(tf.all_variables())
            optimizer = tf.train.AdamOptimizer(lr).minimize(cost) # Adam Optimizer
            sess.run(tf.initialize_all_variables())
            if pre_training:
                sess.run(word_emb.assign(pre_emb_w))
            best_f1 = 0.0
            for e in xrange(nepochs):
                # shuffle
                shuffle([train_lex,train_cue,train_y], 20)
                # TRAINING STEP
                train_tot_acc = []
                dev_tot_acc = []
                tic = time.time()
                for i in xrange(len(train_lex)):
                    acc_train = feeder(train_lex[i],train_cue[i],train_y[i])
                    # Calculating batch accuracy
                    train_tot_acc.append(acc_train)
                    print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./len(train_lex)),'completed in %.2f (sec) <<\r'%(time.time()-tic),
                    sys.stdout.flush()               
                print "TRAINING MEAN ACCURACY: ", sum(train_tot_acc)/len(train_lex)
                # DEVELOPMENT STEP
                pred_dev = []
                gold_dev = []
                for i in xrange(len(valid_lex)):
                    acc_dev, pred, Y_dev = feeder(valid_lex[i],valid_cue[i],valid_y[i],train=False)
                    dev_tot_acc.append(acc_dev)
                    pred_dev.append(pred[:len(valid_lex[i])])
                    gold_dev.append(Y_dev[:len(valid_lex[i])])
                print 'DEV MEAN ACCURACY: ',sum(dev_tot_acc)/len(valid_lex)
                f1,rep_dev,cm_dev = get_eval(pred_dev,gold_dev)
                # STORE INTERMEDIATE RESULTS
                if f1 > best_f1:
                    best_f1 = f1
                    print "Best f1 is: ",best_f1
                    be = e
                    # store the model
                    checkpoint_prefix = os.path.join(folder, "model")
                    saver.save(sess, checkpoint_prefix,global_step=be)
                    print "Model saved."
                    write_report(folder,rep_dev,cm_dev,'dev')
                    store_prediction(folder, valid_lex, dic_inv, pred_dev, gold_dev, 'dev')
                    dry = 0
                else:
                    dry += 1

                if abs(be-e) >= 10 and dry>=5:
                    print "Halving the lr..."
                    clr *= 0.5
                    dry = 0

        else:
            # load model from last checkpoint
            checkpoint_file = tf.train.latest_checkpoint(folder)
            saver.restore(sess, checkpoint_file)
            print "Model restored!"
            # Collect the predictions here
            test_tot_acc = []
            pred_test, gold_test = [],[]
            for i in xrange(len(test_lex)):
                acc_test, pred, Y_test = feeder(test_lex[i], test_cue[i], test_y[i], train = False)
                test_tot_acc.append(acc_test)
                # get prediction softmax
                pred_test.append(pred[:len(test_lex[i])])
                gold_test.append(Y_test[:len(test_lex[i])])
            print 'Mean test accuracy: ', sum(test_tot_acc)/len(test_lex)
            _,report_tst,best_test = get_eval(pred_test,gold_test)

            write_report(folder,report_tst,best_test,'test')
            store_prediction(folder, test_lex, dic_inv, pred_test, gold_test, 'test')
