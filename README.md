<h1 align="center">laplace</h1>

<p align="center">
  <a href="https://twitter.com/home?status=Rate%20Prediction%20with%20TensorFlow%20Bidirectional%20RNN%20by%20%40_resotto_%20https://github.com/resotto/laplace"><img src="https://img.shields.io/badge/twitter-tweet-blue.svg"/></a>
  <a href="https://twitter.com/_resotto_"><img src="https://img.shields.io/badge/feedback-@_resotto_-blue.svg" /></a>
  <a href="https://github.com/resotto/laplace/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-GPL%20v3.0-brightgreen.svg" /></a>
</p>

<p align="center">
  Laplace predicts Bitcoin rate with TensorFlow Bidirectional RNN.
</p>

<p align="center">
  <img src="https://raw.github.com/wiki/resotto/laplace/img/demon.png">
</p>

## Getting Started
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
- Predicting Bitcoin rate

## Detail
| Loss | Value |
|:-----|:-------|
| MSE  |9.6013e-4|

## Build
First, let's create input data with public ticker API.  
You could travel around the world for a while.
```bash
python3 create_input_csv.py
```
Second, please convert time units of the data from seconds to minutes.
```bash
python3 convert_to_min.py
```
Finally, please start laplace model learning.
```bash
python3 laplace.py
```
After learning models, you also can check TensorBoard.
```bash
tensorboard --logdir .tensorboard/logs/
```

## Feedback
- Report a bug to [Bug report](https://github.com/resotto/laplace/issues/1).
- [Tweet me](https://twitter.com/_resotto_) with any other feedback.

## License
[GNU General Public License v3.0](https://github.com/resotto/laplace/blob/master/LICENSE).
