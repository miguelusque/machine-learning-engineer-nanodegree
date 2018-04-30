# Machine Learning Engineer Nanodegree
## Capstone Proposal
Miguel Ángel Martínez Fernández

April 6, 2018

### NBA Players positions

In this project, it will be analysed a dataset containing data on NBA Players game stats from 1980 to 2017. One goal of this project is to find the set of features that best describes a player position.

The original dataset for this project has been taken from this repository at [Kaggle](https://www.kaggle.com/drgilermo/nba-players-stats/data), which in turn has scrapped it from [Basketball-reference](https://www.basketball-reference.com).

A pre-processed version of this dataset is included in this folder, in a file named 'seasons_stats.csv'. That will be our starting dataset for this project.

A model will be trained to predict player positions based on their stats for that set of features. This model could be used by NBA trainers to rethink his players’ position having into consideration their last year stats. Some players, when they get older, move their playing position to more interior roles, to compensate the loss of velocity.

Machine learning has been previously used to make sports predictions. In the following [link](https://www.sciencedirect.com/science/article/pii/S2210832717301485), it can be found a critical survey of the literature on ML for sports result prediction, focusing on the use of neural networks (NN) for this problem.

### Technical Requirements

This project has been developed in [Python 3.6.5](https://www.python.org/downloads/release/python-365/), on a [Jupyter Notebook](http://jupyter.org/), by making use of the following libraries: 
- [IPython](https://ipython.org/)
- [Keras](http://keras.io/)
- [Matplotlib](https://matplotlib.org/)
- [NumPy](http://www.numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [TensorFlow](http://tensorflow.org/)
- [scikit-learn](http://scikit-learn.org/)

For the benefit of the reader, a functional description has been added prior to each code section. Nevertheless, some basic [Python](https://www.python.org/) and [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning) is recommended for a complete understanding of the project.

### Important: Other people's work acknowledgment
Please notice that the content of the file `visuals.py` is almost identical to the file used in different sample projects of this nanodegree. I would like to mention this to do not violate the Honor Code.

In case you are not used to this file, it contains a couple of helper methods to visualize data. I have slightly modified them to fit this project needs.
