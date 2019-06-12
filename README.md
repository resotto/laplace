<h1 align="center">laplace</h1>

<p align="center">
  <a href="https://twitter.com/home?status=Rate%20Prediction%20with%20TensorFlow%20Bidirectional%20RNN%20by%20%40_resotto_%20https://github.com/resotto/laplace"><img src="https://img.shields.io/badge/twitter-tweet-blue.svg"/></a>
  <a href="https://twitter.com/_resotto_"><img src="https://img.shields.io/badge/feedback-@_resotto_-blue.svg" /></a>
  <a href="https://github.com/resotto/laplace/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg" /></a>
</p>

<p align="center">
  Laplace predicts ticker rate with TensorFlow Bidirectional RNN.
</p>

<p align="center">
  <img src="https://raw.github.com/wiki/resotto/laplace/img/demon.png">
</p>

## Getting Started
Please install python3, NumPy, TensorFlow, and scikit-learn(sklearn) in advance.
```bash
python3 --version
pip3 show numpy
pip3 show tensorflow
pip3 show sklearn
```

After confirming your installation, let's get started!
```bash
git clone git@github.com:resotto/laplace.git
cd laplace
python3
```
```python
>>> import laplace as la
>>> input = la.make_temp_input_data()
>>> predicted = la.predict(input)
>>> predicted                         # following values are examples
array([962064.7   , 962072.6   , 962062.94  ,   2000.8098], dtype=float32)
```

## Features
- Predicting ["sell", "buy", "last", "vol"] values of ticker

  - Predicted values are 10 minutes after the last input data (adjustable).
  - Laplace has already been learned with `input_min.csv`, which includes values of BTCJPY ticker.
  - **NOTE：If you want to predict other types of ticker like BTCUSD, please follow [Build instruction](#build).**

## Model Detail


- Forward hidden layer is `tf.keras.layers.LSTMCell`.
- Backward hidden layer is also `tf.keras.layers.LSTMCell`.
- Entire hidden layer is `tf.nn.static_bidirectional_rnn`.
- Optimizer is `Adam`.
- Loss is calculated by `MSE`. Final value of loss is below:

| Loss |  Value  |
|:-----|:--------|
| MSE  |9.6013e-4|



## Build
First, let's create input data with public ticker API.  
You can change the URL of public ticker API in `create_input_csv.py`.
```python
URL = 'https://public.bitbank.cc/btc_jpy/ticker' # Please change this url as you like
```
After changing url, you also need to fix those parts in `create_input_csv.py`:
```python
L13: with open(PATH, 'a') as f: # After changing above url, you also need to fix header below

L37: f.write('time,sell,buy,high,low,last,vol\n') # After changing above url, you also need to fix header below

L50: data = body['data'] # After changing above url, you also need to fix this depending on your url
```
Now you start fetching.

```bash
python3 create_input_csv.py
```
Second, please convert time units of the data from seconds to minutes.  
If you changed ticker url, you also need to fix those parts in `convert_to_min.py`:
```python
L16: f.write('time,sell,buy,high,low,last,vol\n') # please change this header depending on ticker

L24: if pattern.match(row[0]): # row[0] equals "time"
```
Now you convert the data.
```bash
python3 convert_to_min.py
```
Before learning model of laplace, you can adjust parameters below:
```python
MAXLEN = 41   # Time length of input data
INTERVAL = 10 # distance between the last input value and answer value
N_IN = 4      # which means [sell, buy, last, vol]
N_HIDDEN = 13 # Number of hidden layers
N_OUT = 4     # which means [sell, buy, last, vol]
PATIENCE = 10 # Max step of EarlyStopping

L60: optimizer = tf.train.AdamOptimizer(learning_rate=0.0015, beta1=0.9, beta2=0.999)↲
```
If you changed ticker url, you also need to fix those parts in `laplace.py`:
```python
L98: del row[0]   # exclude "time"
L99: del row[2:4] # exclude "high", "low"
L129: diff_aq = np.delete(a - q[-1], -1, 0) # exclude "vol" column
L135: diff_pq = np.delete(p - q[-1], -1, 0) # exclude "vol" column
L146: diff_qp = np.delete(q[-1] - p, -1, 0) # exclude "vol" column
```
Finally, please start laplace model learning.
```bash
python3 laplace.py
```
After learning models, you also can check TensorBoard.
```bash
tensorboard --logdir .tensorboard/logs/
```
When predicting, please follow [Getting Started](#getting-started).

## Feedback
- Report a bug to [Bug report](https://github.com/resotto/laplace/issues/1).
- [Tweet me](https://twitter.com/_resotto_) with any other feedback.

## License
[GNU General Public License v3.0](https://github.com/resotto/laplace/blob/master/LICENSE).
