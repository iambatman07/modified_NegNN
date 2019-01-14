# -*-coding:utf-8-*-

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from NegNN.utils.tools import shuffle, padding, random_uniform, unpickle_data
from NegNN.utils.metrics import *
from NegNN.processors import int_processor, ext_processor
import numpy
import random
import codecs
import os,sys
import time
import subprocess

def _bilstm(scope_dect,
    event_dect,
    tr_lang,
    folder,
    clr,
    n_hidden,
    n_classes,
    emb_size,
    nepochs,
    max_sent_len,
    POS_emb,
    pre_training,
    update,
    training,
    test_files,
    test_lang):
    
    if training:
        # load data
        if not pre_training:
            train_set, valid_set, voc, dic_inv = int_processor.load_train_dev(scope_dect, event_dect, tr_lang, folder)
            vocsize = len(voc['w2idxs'])
            tag_voc_size = len(voc['t2idxs']) if POS_emb == 1 else len(voc['tuni2idxs'])
        else:
            train_set, valid_set, dic_inv, pre_emb_w, pre_emb_t = ext_processor.load_train_dev(scope_dect, event_dect, tr_lang, emb_size, POS_emb)
            vocsize = pre_emb_w.shape[0]
            tag_voc_size = pre_emb_t.shape[0]

        train_lex, train_tags, train_tags_uni, train_cue, _, train_y = train_set
        valid_lex, valid_tags, valid_tags_uni, valid_cue, _, valid_y = valid_set
        print train_lex[2],train_tags[2]
        print train_lex[20],train_tags[20]
    else:      
        # Load data
        if not pre_training:
            assert test_lang == tr_lang
            _, _, voc, dic_inv = unpickle_data(folder)
            test_lex, test_tags, test_tags_uni, test_cue, _, test_y = int_processor.load_test(test_files, voc, scope_dect, event_dect, test_lang)
        else:
            test_set, dic_inv, pre_emb_w, pre_emb_t = ext_processor.load_test(test_files, scope_dect, event_dect, test_lang, emb_size, POS_emb)
            test_lex, test_tags, test_tags_uni, test_cue, _, test_y = test_set

        if pre_training:
            vocsize = pre_emb_w.shape[0]
            tag_voc_size = pre_emb_t.shape[0]
        else:
            vocsize = len(voc['w2idxs'])
            tag_voc_size = len(voc['t2idxs']) if POS_emb == 1 else len(voc['tuni2idxs'])

# **************************************************************************

    # tf placeholders
    seq_len = tf.placeholder(tf.int64, name="seq_len")
    lr = tf.placeholder(tf.float32, name="lr")
    x = tf.placeholder(tf.int32, name="input_x")
    c = tf.placeholder(tf.int32, name="input_c")
    t = tf.placeholder(tf.int32, name="input_t")
    mask = tf.placeholder("float", name="mask")
    istate_fw = tf.placeholder("float", [None, 2*n_hidden],name="istate_fw")
    istate_bw = tf.placeholder("float", [None, 2*n_hidden],name="istate_bw")
    y = tf.placeholder("float", [None, n_classes],name="y")

    # Define weights
    with tf.device("/cpu:0"):
        _weights = {
            'w_emb' : random_uniform([vocsize, emb_size],'we',update),
            't_emb' : random_uniform([tag_voc_size,emb_size],'te',update)
            }

    _weights.update({
        'c_emb' : random_uniform([3,emb_size],'ce'),
        'hidden_w': random_uniform([emb_size, 2*n_hidden],'hidden_w'),
        'hidden_c': random_uniform([emb_size, 2*n_hidden],'hidden_c'),
        'hidden_t': random_uniform([emb_size, 2*n_hidden],'hidden_t'),
        'out_w': random_uniform([2*n_hidden, n_classes],'out_w')
    })
    _biases = {
        'hidden_b': tf.Variable(tf.random_normal([2*n_hidden]),name='hidden_b'),
        'out_b': tf.Variable(tf.random_normal([n_classes]),name="out_b")
    }

    def BiLSTM(_X, _C, _T, _istate_fw, _istate_bw, _weights, _biases):
        # input: a [len_sent,len_seq] (e.g. 7x5)
        # transform into embeddings
        with tf.device("/cpu:0"):
            emb_x = tf.nn.embedding_lookup(_weights['w_emb'],_X)
            emb_t = tf.nn.embedding_lookup(_weights['t_emb'],_T)
            emb_c = tf.nn.embedding_lookup(_weights['c_emb'],_C)

        # Linear activation
        _X = tf.matmul(emb_x, _weights['hidden_w']) + tf.matmul(emb_c, _weights['hidden_c']) + tf.matmul(emb_t,_weights['hidden_t']) + _biases['hidden_b']

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(0,max_sent_len,_X)

        # Get lstm cell output
        outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, _X,initial_state_fw = _istate_fw, initial_state_bw=_istate_bw,sequence_length = seq_len)

        return outputs

    pred = BiLSTM(x, c, t, istate_fw, istate_bw, _weights, _biases)

    last_y = [tf.matmul(item, _weights['out_w']) + _biases['out_b'] for item in pred]
    final_outputs = tf.squeeze(tf.pack(last_y))

    # Define cost
    cost = tf.reduce_sum(tf.mul(tf.nn.softmax_cross_entropy_with_logits(final_outputs, y),mask))/tf.reduce_sum(mask) # softmax

    predictions = tf.nn.softmax(final_outputs)

    accuracy = tf.reduce_sum(tf.mul(tf.cast(tf.equal(tf.argmax(predictions,1), tf.argmax(y,1)), "float"),mask))/tf.reduce_sum(mask)

# *****************************************************************

    def feeder(lex, cue, tags, _y, train = True):
        X = padding(lex, max_sent_len, vocsize - 1)
        C = padding(cue, max_sent_len, 2)
        T = padding(tags, max_sent_len, tag_voc_size - 1)
        Y = padding(numpy.asarray(map(lambda x: [1,0] if x == 0 else [0,1],_y)).astype('int32'),max_sent_len,0,False)
        _mask = [1 if token!=vocsize - 1 else 0 for token in X]
        feed_dict={
            x: X,
            c: C,
            t: T,
            y: Y,
            istate_fw: numpy.zeros((1, 2*n_hidden)),
            istate_bw: numpy.zeros((1, 2*n_hidden)),
            seq_len: numpy.asarray([len(lex)]),
            mask: _mask}
        if train:
            feed_dict.update({lr:clr})
            _, acc_train = sess.run([optimizer, accuracy], feed_dict = feed_dict)
            return acc_train
        else:
            acc_test, pred = sess.run([accuracy,predictions], feed_dict = feed_dict)
            return acc_test, pred , Y

    saver = tf.train.Saver()
    # Launch the session  
    with tf.Session() as sess:
        if training:
            saver = tf.train.Saver(tf.all_variables())
            optimizer = tf.train.AdamOptimizer(lr).minimize(cost) # Adam Optimizer
            sess.run(tf.initialize_all_variables())
            if pre_training:
                sess.run(_weights['w_emb'].assign(pre_emb_w))
                sess.run(_weights['t_emb'].assign(pre_emb_t))
            best_f1 = 0.0
            for e in xrange(nepochs):
                # shuffle
                shuffle([train_lex, train_tags, train_tags_uni, train_cue, train_y], 20)
                # TRAINING STEP
                train_tot_acc = []
                dev_tot_acc = []
                tic = time.time()
                for i in xrange(len(train_lex)):
                    acc_train = feeder(train_lex[i],train_cue[i],train_tags[i] if POS_emb == 1 else train_tags_uni[i], train_y[i])
                    # Calculating batch accuracy
                    train_tot_acc.append(acc_train)
                    print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./len(train_lex)),'completed in %.2f (sec) <<\r'%(time.time()-tic),
                    sys.stdout.flush()               
                print "TRAINING MEAN ACCURACY: ", sum(train_tot_acc)/len(train_lex)
                # DEVELOPMENT STEP
                pred_dev = []
                gold_dev = []
                for i in xrange(len(valid_lex)):
                    acc_dev, pred, Y_dev = feeder(valid_lex[i],valid_cue[i],valid_tags[i] if POS_emb == 1 else valid_tags_uni[i],valid_y[i],train=False)
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
            saver.restore(sess,checkpoint_file)
            print "Model restored!"
            # Collect the predictions here
            test_tot_acc = []
            preds_test, gold_test = [],[]
            for i in xrange(len(test_lex)):
                acc_test, pred_test, Y_test = feeder(test_lex[i], test_cue[i], test_tags[i] if POS_emb == 1 else test_tags_uni[i],test_y[i], train = False)
                test_tot_acc.append(acc_test)
                # get prediction softmax
                preds_test.append(pred_test[:len(test_lex[i])])
                gold_test.append(Y_test[:len(test_lex[i])])
            print 'Mean test accuracy: ', sum(test_tot_acc)/len(test_lex)
            _,report_tst,best_test = get_eval(preds_test,gold_test)

            write_report(folder,report_tst,best_test,'test')
            store_prediction(folder, test_lex, dic_inv, preds_test, gold_test, 'test')                    
