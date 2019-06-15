import os
import csv
import re
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

np.random.seed(0)
tf.set_random_seed(1234)

INPUT_DATA = 'input_min.csv'
LOG_DIR    = '.tensorboard/logs'
MODEL_DIR  = '.model'
CHECKPOINT = '/model.ckpt'
SCALER     = '/scaler.save'

MAXLEN           = 41                                     # Time series length of input data
INTERVAL         = 10                                     # Time interval between the last input value and answer value
N_IN             = 4                                      # which means [sell, buy, last, vol]
N_HIDDEN         = 13                                     # Number of hidden layers
N_OUT            = 4                                      # which means [sell, buy, last, vol]
LEARNING_RATE    = 0.0015                                 # Optimizer's learning rate
PATIENCE         = 10                                     # Max step of EarlyStopping
INPUT_VALUE_TYPE = ['sell', 'buy', 'last', 'vol']         # Input value type
EPOCHS           = 1000                                   # Epochs
BATCH_SIZE       = 50                                     # Batch size


def inference(x, n_in=None, maxlen=None, n_hidden=None, n_out=None):
    def weight_bariable(shape, name=None):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial, name=name)

    def bias_variable(shape, name=None):
        initial = tf.zeros(shape, dtype=tf.float32)
        return tf.Variable(initial, name=name)

    # In order to adjust to specification of tf.nn.static_bidirectional_rnn,
    # reshape format of recurrent data to (batch_size, input_dim)
    x = tf.transpose(x, [1, 0, 2]) # Tensor: (?, MAXLEN, N_IN) => Tensor: (MAXLEN, ?, N_IN)
    x = tf.reshape(x, [-1, n_in])  # Tensor: (MAXLEN, ?, N_IN) => Tensor: (?, N_IN)
    x = tf.split(x, maxlen, 0)     # Tensor: (?, N_IN)         =>   list: len(x): MAXLEN

    cell_forward = layers.LSTMCell(n_hidden, unit_forget_bias=True)
    cell_backward = layers.LSTMCell(n_hidden, unit_forget_bias=True)
    outputs, _, _ = tf.nn.static_bidirectional_rnn(cell_forward, cell_backward, x, dtype=tf.float32)

    W = weight_bariable([n_hidden * 2, n_out], name='W')
    tf.summary.histogram('W', W) # TensorBoard
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
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.999)
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
    # csv check
    if not os.path.exists(INPUT_DATA):
        print('please prepare input data first:', INPUT_DATA)
        exit()

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
    answer checking

    checking correct answer with q(question), p(prediction), and a(answer).

    I. increasing case (a - q[-1] >= 0)

        if p - q[-1] >= 0:
            correct_counts += 1

    II. decreasing case (a - q[-1] < 0)

        if p - q[-1] < 0:
            correct_counts += 1
    '''

    if verbose:
        print('-' * 10)
        print('question: ')
        print(q)
        print('prediction: ', p)
        print('answer:     ', a)

    diff_aq = a - q[-1]
    if verbose:
        print('diff between answer and question:     ', diff_aq)
    diff_aq = diff_aq >= 0

    diff_pq = p - q[-1]
    if verbose:
        print('diff between prediction and question: ', diff_pq)
    diff_pq = diff_pq >= 0

    if verbose:
        diff_ap = a - p
        print('diff between answer and prediction:   ', diff_ap)

    correct_counts = np.zeros(N_OUT)

    for i in range(len(correct_counts)):
        if diff_aq[i]:          # increasing case
            if diff_pq[i]:
                if verbose:
                    print(INPUT_VALUE_TYPE[i], 'is correct answer!')
                correct_counts[i] += 1
        else:                   # decreasing case
            if not diff_pq[i]:
                if verbose:
                    print(INPUT_VALUE_TYPE[i], 'is correct answer!')
                correct_counts[i] += 1

    return correct_counts


def check_accuracy(correct_counts, number_of_tests, epoch):
    accuracy = correct_counts * 100 / number_of_tests
    if epoch >= 10:
        print('accuracy per 10 epoch:           {}'.format(accuracy))
    return accuracy


def check_accuracy_average(val_acc_ave, acc, length=10):
    val_acc_ave.insert(0, acc)
    if len(val_acc_ave) > length:
        val_acc_ave.pop()
    if len(val_acc_ave) == length:
        accuracy_total = np.zeros(N_OUT)
        for i in range(len(val_acc_ave)):
            accuracy_total += val_acc_ave[i]
        accuracy_total /= length
        print('average of the last {} accuracy: {}'.format(length, accuracy_total))


def predict(arr_f):
    tf.reset_default_graph()

    # type check
    if not isinstance(arr_f, np.ndarray):
        print('input data has to be numpy ndarray')
        return

    # shape check
    if arr_f.ndim != 2 or arr_f.shape[0] != MAXLEN or arr_f.shape[1] != N_IN:
        print('input data is numpy array whose shape is (', MAXLEN, ',', N_IN, '),')
        print('which means values of', INPUT_VALUE_TYPE, ' x', MAXLEN, ' rows')
        return

    # learning check
    if not os.path.exists(MODEL_DIR + SCALER):
        print('model is not learned yet. please learn first')
        return

    # normalization
    scaler = joblib.load(MODEL_DIR + SCALER)
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
    arr_f = arr_f.reshape(1, MAXLEN, N_IN)

    sess = tf.Session()

    # restoring variables
    tf.train.Saver().restore(sess, MODEL_DIR + CHECKPOINT)

    prediction = y.eval(session=sess, feed_dict={
        x: arr_f
    })
    np.set_printoptions(suppress=True)

    # restoring values from normalization
    p = scaler.inverse_transform(prediction)

    return p[0]


def predict_rising_from(arr_f):
    tmp = predict(arr_f)
    predicted = (tmp - arr_f[-1]) > 0
    return predicted


def predict_falling_from(arr_f):
    risingPrediction = predict_rising_from(arr_f)
    return np.logical_not(risingPrediction)


def make_input_data():
    list = get_input_data()
    tmp_idx = np.random.randint(0, len(list) - MAXLEN + 1)
    list = list[tmp_idx:tmp_idx + MAXLEN]
    return np.array(list)


if __name__ == '__main__':
    # TensorBoard directory check
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # model directory check
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    '''
    producing data
    '''
    f = get_input_data() # input data list

    # normalization
    scaler = MinMaxScaler(feature_range=(-1, 1))
    arr_f = scaler.fit_transform(np.array(f))
    joblib.dump(scaler, MODEL_DIR + SCALER)

    data = []
    target = []

    # + 1 is caused by exclusiveness of stop value of range
    for i in range(0, len(f) - MAXLEN - INTERVAL + 1):
        data.append(arr_f[i: i + MAXLEN])
        target.append(arr_f[i + MAXLEN + INTERVAL - 1])

    X = np.array(data).reshape(len(data), MAXLEN, N_IN)
    Y = np.array(target).reshape(len(data), N_OUT)
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
    epochs = EPOCHS
    batch_size = BATCH_SIZE
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

        '''
        testing
        '''
        if epoch % 10 == 0:
            print('-' * 10)
            correct_counts = np.zeros(N_OUT)

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
                p = scaler.inverse_transform(prediction)[index]
                a = scaler.inverse_transform(answer)[index]

                correct_counts += check_answer(q=q, a=a, p=p, verbose=1)

            accuracy = check_accuracy(correct_counts, number_of_tests, epoch)
            check_accuracy_average(val_acc_ave, accuracy)

        print()
        print('epoch:', epoch, ', validation loss:', val_loss)

    # model saving
    model_path = saver.save(sess, MODEL_DIR + CHECKPOINT)
    print('Model saved to:', model_path)

