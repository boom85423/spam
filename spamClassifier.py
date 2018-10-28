'''
title: "Spam Recognition"
author: "Yi-Zhan Xu"
date: "2018/10/26"
'''
import io, sys, os, glob
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import tensorflow as tf

sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.set_random_seed(1)
np.random.seed(1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default="Youtube01-Psy.csv")
    parser.add_argument('--model_path', type=str, default="D2V/doc2vec")
    parser.add_argument('--params_path', type=str, default="DNN/params")
    parser.add_argument('--batch_size', type=int, default=32) 
    parser.add_argument('--verbose', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=500) 
    parser.add_argument('--lr', type=int, default=5e-3) 
    parser.add_argument('--dropout', type=int, default=0.5)
    parser.add_argument('--text', type=str, default="I am spam")
    parser.add_argument('--visual', type=str, default="visual/")
    parser.add_argument('--status', type=str, default="train")
    return parser.parse_args()

def preprocessing(data_path):
    data = pd.read_csv(data_path, encoding="utf-8")
    labels = data["CLASS"].values
    labels_onehot = []
    for i in labels:
        if i == 0:
            labels_onehot.append([0, 1])
        else:
            labels_onehot.append([1, 0])
    labels_onehot = np.vstack(labels_onehot)
    content = data["CONTENT"].values
    content_list = []
    content_length = []
    for i in content:
        i = i.replace("<br />", "")
        words = i.split()
        words[-1] = words[-1].replace("\ufeff", "")
        content_length.append(len(words))
        content_list.append(words)
    vector_size = np.max(content_length)
    return content_list, labels_onehot, vector_size

def D2V(content_list, tag, vector_size, model_path):
    content_meta = [TaggedDocument(content_list[i], ["%s_%d" % (tag,i)]) for i in range(len(content_list))]
    model = Doc2Vec(content_meta, vector_size=vector_size)
    model.train(content_meta, total_examples=model.corpus_count, epochs=100)
    model.save(model_path)
    model = Doc2Vec.load(args.model_path)
    content_vec = [model.docvecs["%s_%d" % (tag,i)] for i in range(len(content_list))]
    content_vec = np.asarray(content_vec)
    return content_vec

def next_batch(x, y, batch_size):
    idx = np.arange(0, x.shape[0])
    np.random.shuffle(idx)
    x_shuffle = x[idx]
    y_shuffle = y[idx]
    return x_shuffle[0:batch_size], y_shuffle[0:batch_size]

def demo(text):
    with tf.Session() as sess: 
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        saver.restore(sess, args.params_path)
        model = Doc2Vec.load(args.model_path)
        text_vec = model.infer_vector(args.text).reshape(1,-1)
        test_output = sess.run(output, {tf_x:text_vec})  
        pred_y = np.argmax(test_output, 1)[0]
    if pred_y == 1:
        print("## %s -> SPAM" % text)
    else:
        print("## %s -> HAM" % text)

def infer_vec(data):
    content_list, labels_onehot, vector_size = preprocessing("comment/" + data)
    model = Doc2Vec.load(args.model_path)
    content_vec = np.asarray([model.infer_vector(args.text) for i in content_list])
    return content_vec, labels_onehot

if __name__ == "__main__":
    args = parse_args()
    content_list, labels_onehot, vector_size = preprocessing("comment/" + args.train_data)
    content_list = D2V(content_list, "embedding", vector_size, args.model_path)
    x_train, x_test, y_train, y_test = train_test_split(content_list, labels_onehot, test_size=0.2)

    tf_x = tf.placeholder(tf.float32, [None,vector_size])
    tf_y = tf.placeholder(tf.float32, [None,2])

    l1 = tf.layers.dense(tf_x, vector_size*1.5, tf.nn.relu)
    l1_dropout = tf.layers.dropout(inputs=l1, rate=args.dropout)
    l2 = tf.layers.dense(l1_dropout, vector_size**0.5, tf.nn.relu)
    l2_dropout = tf.layers.dropout(inputs=l2, rate=args.dropout)
    output = tf.layers.dense(l2_dropout, 2)

    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf_y, logits=output)
    train_op = tf.train.AdamOptimizer(args.lr).minimize(loss) 
    accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

    with tf.Session() as sess: # use the data to train
        print("## Use %s to train..." % args.train_data)
        writer = tf.summary.FileWriter(args.visual + args.status, sess.graph)
        tf.summary.scalar("LOSS", loss)
        merge_op = tf.summary.merge_all()
        
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver() 

        for epoch in range(args.epochs):
            for step in range(round(x_train.shape[0] / args.batch_size)):
                b_x, b_y = next_batch(x_train, y_train, args.batch_size)
                train_, ACC = sess.run([train_op, accuracy], {tf_x:b_x, tf_y:b_y})
                if args.status == "train": # early stopping
                    feed_dict = {tf_x:b_x, tf_y:b_y}
                else:
                    feed_dict = {tf_x:x_test, tf_y:y_test}
                loss_, res_ = sess.run([loss, merge_op], feed_dict=feed_dict)
            writer.add_summary(res_, epoch)                        
            if epoch % args.verbose == 0:
                print("Epoch: %d, ACC_train: %.4f" % (epoch, ACC))           
        saver.save(sess, args.params_path)

    with tf.Session() as sess: # reloda to test same data
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        saver.restore(sess, args.params_path)
        ACC_test = sess.run([accuracy], {tf_x:x_test, tf_y:y_test})
        print("## ACC_test: %.4f" % ACC_test[0])

    demo(args.text)

    for csv in glob.glob(os.path.join("comment/", "*.csv")): # test another 
        if csv != "comment/" + args.train_data:
            with tf.Session() as sess:
                init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(init_op)
                saver = tf.train.Saver()
                saver.restore(sess, args.params_path)
                content_vec, labels_onehot = infer_vec(csv.replace("comment/",""))
                ACC_test = sess.run([accuracy], {tf_x:content_vec, tf_y:labels_onehot})
                print("## ACC_test_%s: %.4f" % (csv.replace("comment/","") ,ACC_test[0]))