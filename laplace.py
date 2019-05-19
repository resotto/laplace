import os
import csv
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

np.random.seed(0)
tf.set_random_seed(1234)

INPUT_DATA = 'input_min.csv'
LOG_DIR = '.tensorboard/logs'
MODEL_DIR = '.model'
CHECKPOINT = '/model.ckpt'

MAXLEN = 41
INTERVAL = 10 # distance between the last input value and answer value
N_IN = 4      # which means [sell, buy, last, vol]
N_HIDDEN = 13
N_OUT = 4     # which means [sell, buy, last, vol]
PATIENCE = 10

def inference(x, n_in=None, maxlen=None, n_hidden=None, n_out=None):
    def weight_bariable(shape, name=None):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial, name=name)

    def bias_variable(shape, name=None):
        initial = tf.zeros(shape, dtype=tf.float32)
        return tf.Variable(initial, name=name)

    # In order to adjust to specification of tf.nn.static_bidirectional_rnn,
    # reshape format of recurrent data to (batch_size, input_dim)
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_in])
    x = tf.split(x, maxlen, 0)

    cell_forward = layers.LSTMCell(n_hidden, unit_forget_bias=True)
    cell_backward = layers.LSTMCell(n_hidden, unit_forget_bias=True)
    outputs, _, _ = tf.nn.static_bidirectional_rnn(cell_forward, cell_backward, x, dtype=tf.float32)

    W = weight_bariable([n_hidden * 2, n_out], name='W')
    b = bias_variable([n_out], name='b')
    y = tf.matmul(outputs[-1], W) + b
    return y


def loss(y, t):
    with tf.name_scope('loss'):
        mse = tf.reduce_mean(tf.square(y - t))
        tf.summary.scalar('mse', mse) # TensorBoard
        return mse


def training(loss):
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0015, beta1=0.9, beta2=0.999)
        train_step = optimizer.minimize(loss)
        return train_step


class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self._step = 0
            self._loss = loss
        return False


def get_input_data():
    ticker_data = []
    with open(INPUT_DATA, newline='') as csvfile:

        # check whether header exists or not
        with open(INPUT_DATA, newline='') as tmp:
            header = tmp.readline()
            if re.match('\D+', header):
                print('skipping header')
                next(csvfile)

        reader = csv.reader(csvfile)
        for row in reader:
            del row[0]   # exclude "time"
            del row[2:4] # exclude "high", "low"
            ticker_data.append([float(v) for v in row])
    return ticker_data


def check_answer(q, a, p, verbose=None):
    '''
    accuracy checking

    checking correct answer with q(input), p(prediction), and a(answer).
    "vol" column will be excluded in advance.

    I. increasing case (a - q[-1] >= 0)

        if p - q[-1] >= 0:
            correct_counts += 1

    II. decreasing case (a - q[-1] < 0)

        if q[-1] - p >= 0:
            correct_counts += 1
    '''

    if verbose:
        print('-' * 10)
        print('input: ')
        print(q)
        print('prediction: ', p)
        print('answer:     ', a)

    diff_aq = np.delete(a - q[-1], -1, 0)     # exclude "vol" column
    if verbose:
        print('diff between answer and question:     ', diff_aq)
    diff_aq = diff_aq < 0

    if not np.any(diff_aq): # increasing case
        diff_pq = np.delete(p - q[-1], -1, 0) # exclude "vol" column
        if verbose:
            print('diff between prediction and question: ', diff_pq)
        diff_pq = diff_pq < 0 # don't count if any one of values is True
        if not np.any(diff_pq):
            if verbose:
                print('correct answer!')
            return 1
        else:
            return 0
    else:                   # decreasing case
        diff_qp = np.delete(q[-1] - p, -1, 0) # exclude "vol" column
        if verbose:
            print('diff between question and prediction: ', diff_qp)
        diff_qp = diff_qp < 0 # don't count if any one of values is True
        if not np.any(diff_qp):
            if verbose:
                print('correct answer!')
            return 1
        else:
            return 0


def check_accuracy(correct_counts, number_of_tests, epoch):
    accuracy = correct_counts * 100 / number_of_tests
    if epoch >= 10:
        print('accuracy per 10 epoch:           {:.1f}'.format(accuracy), '%')
    return accuracy


def check_accuracy_average(val_acc_ave, acc, length=10):
    val_acc_ave.insert(0, acc)
    if len(val_acc_ave) > length:
        val_acc_ave.pop()
    if len(val_acc_ave) == length:
        accuracy_total = 0
        for i in range(len(val_acc_ave)):
            accuracy_total += val_acc_ave[i]
        print('average of the last {} accuracy: {:.1f} {}'.format(length, accuracy_total / length, '%'))


def predict(arr_f):
    tf.reset_default_graph()

    # shape check
    if arr_f.ndim != 2 or arr_f.shape[0] != 41 or arr_f.shape[1] !=  4:
        print('input data is numpy array whose shape is (41, 4),')
        print('which means values of [sell, buy, last, vol] x 41 rows')
        return

    # normalization
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(arr_f)
    arr_f = scaler.transform(arr_f)


    '''
    model setting
    '''
    x = tf.placeholder(tf.float32, shape=[None, MAXLEN, N_IN])
    t = tf.placeholder(tf.float32, shape=[None, N_OUT])
    y = inference(x, n_in=N_IN, maxlen=MAXLEN, n_hidden=N_HIDDEN, n_out=N_OUT)


    '''
    prediction
    '''

    # reshape
    arr_f = arr_f.reshape(1, 41, 4)

    sess = tf.Session()
    tf.train.Saver().restore(sess, MODEL_DIR + CHECKPOINT) # restoring variables
    prediction = y.eval(session=sess, feed_dict={
        x: arr_f
    })

    # restoring values from normalization
    np.set_printoptions(suppress=True)
    p = scaler.inverse_transform(prediction)

    return p[0]


def make_temp_input_data():
    list = get_input_data()
    tmp_idx = np.random.randint(0, len(list) - MAXLEN + 1)
    list = list[tmp_idx:tmp_idx + MAXLEN]
    return np.array(list)


if __name__ == '__main__':
    '''
    TensorBoard directory check
    '''
    if os.path.exists(LOG_DIR) is False:
        os.makedirs(LOG_DIR)


    '''
    model directory check
    '''
    if os.path.exists(MODEL_DIR) is False:
        os.mkdir(MODEL_DIR)


    '''
    producing data
    '''
    f = get_input_data() # input data list

    # normalization
    scaler = MinMaxScaler(feature_range=(-1, 1))
    arr_f = np.array(f)
    scaler.fit(arr_f)
    arr_f = scaler.transform(arr_f)

    length_of_input = len(f) # input data length

    data = []
    target = []

    # + 1 is caused by exclusiveness of stop value of range
    for i in range(0, length_of_input - MAXLEN - INTERVAL + 1):
        data.append(arr_f[i: i + MAXLEN])
        target.append(arr_f[i + MAXLEN + INTERVAL - 1])

    X = np.array(data).reshape(len(data), MAXLEN, 4)
    Y = np.array(target).reshape(len(data), 4)
    N_train = int(len(data) * 0.9)
    N_validation = len(data) - N_train
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=N_validation)


    '''
    model setting
    '''
    x = tf.placeholder(tf.float32, shape=[None, MAXLEN, N_IN])
    t = tf.placeholder(tf.float32, shape=[None, N_OUT])
    y = inference(x, n_in=N_IN, maxlen=MAXLEN, n_hidden=N_HIDDEN, n_out=N_OUT)
    loss = loss(y, t)
    train_step = training(loss)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=1)


    '''
    learning
    '''
    epochs = 1000
    batch_size = 50
    np.set_printoptions(suppress=True)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver() # for saving model
    sess = tf.Session()

    # TensorBoard
    file_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    summaries = tf.summary.merge_all()

    sess.run(init)

    n_batches = N_train // batch_size
    val_acc_ave = []
    number_of_tests = N_validation * 10 // epochs
    validation_indices = [v for v in range(N_validation)]

    for epoch in range(epochs):

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            sess.run(train_step, feed_dict={
                x: X_train[start:end],
                t: Y_train[start:end]
            })

        # TensorBoard
        summary, val_loss = sess.run([summaries, loss], feed_dict={
            x: X_validation,
            t: Y_validation
        })
        file_writer.add_summary(summary, epoch)

        if early_stopping.validate(val_loss):
            break

        # testing
        if epoch % 10 == 0:
            print('-' * 10)
            correct_counts = 0

            for i in range(number_of_tests):
                tmp = np.random.randint(0, len(validation_indices))
                index = validation_indices.pop(tmp)

                question = X_validation
                answer = Y_validation
                prediction = y.eval(session=sess, feed_dict={
                    x: question
                })

                # restoring values from normalization
                q = scaler.inverse_transform(question[index])
                p = scaler.inverse_transform(prediction)
                a = scaler.inverse_transform(answer)
                p = p[index]
                a = a[index]

                correct_counts += check_answer(q=q, a=a, p=p, verbose=0)

            accuracy = check_accuracy(correct_counts, number_of_tests, epoch)
            check_accuracy_average(val_acc_ave=val_acc_ave, acc=accuracy)

        print()
        print('epoch:', epoch, ', validation loss:', val_loss)

    # model saving
    model_path = saver.save(sess, MODEL_DIR + CHECKPOINT)
    print('Model saved to:', model_path)

