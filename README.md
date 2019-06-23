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
cd laplace/btcusd
python3
```
```python
>>> import laplace as la
>>> input = la.make_input_data()

>>> type(input)
<class 'numpy.ndarray'>

>>> input.shape
(41, 4)

>>> predicted = la.predict(input)
>>> predicted                         # following values are examples
array([ 9191.143,  9191.745,  9191.728, 19837.059], dtype=float32)

>>> rising = la.predict_rising_from(input)
>>> rising                            # following values are examples
array([False, False, False, False])

>>> falling = la.predict_falling_from(input)
>>> falling                           # following values are examples
array([ True,  True,  True,  True])
```

## Features
- Predicting ticker values
- Predicting rising of ticker values with boolean
- Predicting falling of ticker values with

## Loss & Accuracy
- Final loss value:

| Loss |  Value  |
|:-----|:--------|
| MSE  |9.807e-4|

- Final average of the last 10 accuracy(%):

| bid |  ask  | last_price | volume |
|:---:|:-----:|:----------:|:------:|
| 85  |  85   | 80         | 50     |


## Details
- Predicted values are 10 minutes after the last input data (adjustable).
- Input data is the past 41 minutes ticker value (adjustable).
- Input dimension and output dimension are 4 (adjustable).
- Accuracy is calculated per 10 epochs (adjustable).


- Forward hidden layer is `tf.keras.layers.LSTMCell`.
- Backward hidden layer is `tf.keras.layers.LSTMCell`.
- Entire hidden layer is `tf.nn.static_bidirectional_rnn`.
- Optimizer is `Adam`.
- Loss is calculated by `MSE`.


- Model's parameters are saved to `.model`.
- TensorBoard's logs are saved to `.tensorboard/logs`.

## Build
First, let's create input data.  
You can change the URL of public ticker API in `create_csv.py`.
```python
L5: URL    = 'https://api.bitfinex.com/v1/pubticker/btcusd' # Please change this url as you like
```

If you changed URL, you also need to fix those parts in `create_csv.py`:
```python
L7: HEADER = 'time,bid,ask,last_price,volume' # Csv header. After changing above url, you may need to fix this
L44: write(time, body)                        # After changing above url, you also need to fix this depending on ticker response
```

Now, you start fetching.  
After running `create_csv.py`, `input.csv` will be created.
```bash
python3 create_csv.py
```

Second, please convert time units of the data in `input.csv` from seconds to minutes.  
After runnning `convert.py`, `input_min.csv` will be created, which is input data for learning.
```bash
python3 convert.py
```

Third, before learning, you can adjust parameters in `laplace.py`.
```python
MAXLEN           = 41                                     # Time series length of input data
INTERVAL         = 10                                     # Time interval between the last input value and answer value
N_IN             = 4                                      # Input dimension
N_HIDDEN         = 13                                     # Number of hidden layers
N_OUT            = 4                                      # Output dimension
LEARNING_RATE    = 0.0015                                 # Optimizer's learning rate
PATIENCE         = 10                                     # Max step of EarlyStopping
INPUT_VALUE_TYPE = ['bid', 'ask', 'last_price', 'volume'] # Input value type
EPOCHS           = 2500                                   # Epochs
BATCH_SIZE       = 50                                     # Batch size
TESTING_INTERVAL = 10                                     # Test interval

RANDOM_LEARNING_ENABLED = True                            # Index of data determined randomly or not
EARLY_STOPPING_ENABLED  = False                           # Early Stopping enabled or not
```

Finally, please start learning.
```bash
python3 laplace.py
```

After learning model, you also can check TensorBoard.
```bash
tensorboard --logdir .tensorboard/logs/
```
When predicting, please follow [Getting Started](#getting-started).

## Feedback
- Report a bug to [Bug report](https://github.com/resotto/laplace/issues/1).
- [Tweet me](https://twitter.com/_resotto_) with any other feedback.

## License
[GNU General Public License v3.0](https://github.com/resotto/laplace/blob/master/LICENSE).
