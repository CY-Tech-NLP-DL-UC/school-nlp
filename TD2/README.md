# Spam filtering classificator

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
* [Brief steps description](#brief-steps-description)
* [Results](#results)

## General info

Spam filtering project from NLP Class : students are asked to create a from-scratch classificator using Naives Bayes formula.
This classificator learns from a training set of tweets in order to determine whether they are ham or spam messages.
Once it learned, it then tries to make some predictions on a test set (different from the training set) and is also able to give an evaluation of its performances.

## Technologies

This project was created using:

* Python 3.6.9
* Numpy 1.16.4
* Pandas 0.24.2
* NLTK 3.5

## Setup

Use Jupyter notebook to run 'main.ipnyb' or run main.py using ```python3 main.py``` (make sure to have the libraries correctly installed in that case)
If first use, please think about uncommenting the two download lines at the beginning, and recomment them afterwards to gain in efficiency.

## Brief steps description

A dataset is splitted in two distinct parts to train and test the classificator (70% / 30%).
Data is preprocessed to clean words, then a dataframe (from pandas) is created ('knowledge') to count the occurence of each unique word in ham and spam messages.
Then a classificator's object class is called to use this knowledge and make predictions on the test set.
Finally, classificator is auto-evaluated.

## Results

Results are supposed to be always the same since randomness was imposed to be always the same (using ```np.random.seed(0)```) in order to show the results.

Here are the outputs :

```
Number of spam instances: 527.
Number of ham instances: 3388.
Total number of words: 7148.

Acccuracy: 0.9783001808318263

Precision on ham label: 1.0
Precision on spam label: 0.8363636363636363

Recall on ham label: 0.9755932203389831
Recall on spam label: 1.0

F1 score on ham label: 0.9876458476321209
F1 score on spam label: 0.9108910891089108

Execution time : 68.92359495162964
```

Note that execution time could be much more different.

## Data analysis

You could also find how the analysis on data was made for the preprocessing part in 'data_analysis.ipynb' (use Jupyter notebook)