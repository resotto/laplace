<h1 align="center">laplace</h1>

<p align="center">
  <a href="https://twitter.com/home?status=Rate%20Prediction%20with%20TensorFlow%20Bidirectional%20RNN%20by%20%40_resotto_%20https://github.com/resotto/laplace"><img src="https://img.shields.io/badge/twitter-tweet-blue.svg"/></a>
  <a href="https://twitter.com/_resotto_"><img src="https://img.shields.io/badge/feedback-@_resotto_-blue.svg" /></a>
  <a href="https://github.com/resotto/laplace/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg" /></a>
</p>

<p align="center">
  Laplace predicts ticker rate.
</p>

<p align="center">
  <img src="https://raw.github.com/wiki/resotto/laplace/img/demon.png">
</p>

## Getting Started
Please install [TensorFlow](https://www.tensorflow.org/) and [scikit-learn(sklearn)](https://scikit-learn.org/stable/) in advance.
```bash
git clone git@github.com:resotto/laplace.git
cd laplace/btcjpy
python3
```
```python
>>> import laplace as la
>>> input = la.make_input_data()
>>> predicted = la.predict(input)
>>> predicted                         # following values are examples
array([962064.7   , 962072.6   , 962062.94  ,   2000.8098], dtype=float32)

>>> rising = la.predict_rising_from(input)
>>> rising                            # following values are examples
array([False, False, False,  True])

>>> falling = la.predict_falling_from(input)
>>> falling                           # following values are examples
array([ True,  True,  True, False])
```

## Features
- Predicting values of ticker
- Predicting rising of ticker values with boolean
- Predicting falling of ticker values with boolean

  - Predicted values are 10 minutes after the last input data (adjustable).
  - BTCJPY's laplace has already been learned with `input_min.csv`, while BTCUSD's one not.
    **If you use BTCUSD's laplace, please follow [build instruction](#build).**

## Details
- Forward hidden layer is `tf.keras.layers.LSTMCell`.
- Backward hidden layer is also `tf.keras.layers.LSTMCell`.
- Entire hidden layer is `tf.nn.static_bidirectional_rnn`.
- Optimizer is `Adam`.
- Loss is calculated by `MSE`. Final value of loss is below:

| Loss |  Value  |
|:-----|:--------|
| MSE  |9.6013e-4|

- TensorBoard's logs are saved to `.tensorboard/logs`.
- Model's parameters are saved to `.model`.

## Build
If you want to predict other types of ticker like BTCUSD, please following instruction.
```bash
cd -
cd laplace/btcusd
```

First, let's create input data with `create_csv.py`.  
You can change the URL of public ticker API.
```python
L5: URL    = 'https://api.bitfinex.com/v1/pubticker/btcusd' # Please change this url as you like
```

If you changed URL, you also need to fix those parts:
```python
L7: HEADER = 'time,bid,ask,last_price,volume' # Csv header. After changing above url, you may need to fix this
L44: write(time, body)                        # After changing above url, you also need to fix this depending on ticker response
```

Now, you start fetching.
```bash
python3 create_csv.py
```

Second, please convert time units of the data from seconds to minutes.  
```bash
python3 convert.py
```

Before learning model of laplace, you can adjust parameters in `laplace.py`.
```python
MAXLEN           = 41                                     # Time series length of input data
INTERVAL         = 10                                     # Time interval between the last input value and answer value
N_IN             = 4                                      # which means [bid, ask, last_price, volume]
N_HIDDEN         = 13                                     # Number of hidden layers
N_OUT            = 4                                      # which means [bid, ask, last_price, volume]
LEARNING_RATE    = 0.0015                                 # Optimizer's learning rate
PATIENCE         = 10                                     # Max step of EarlyStopping
INPUT_VALUE_TYPE = ['bid', 'ask', 'last_price', 'volume'] # Input value type
EPOCHS           = 1000                                   # Epochs
BATCH_SIZE       = 50                                     # Batch size
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
