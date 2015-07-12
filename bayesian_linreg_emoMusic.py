# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 00:23:12 2015

@author: tpellegrini
"""
from pymc3 import  *
import numpy as np
from utils import load_X, load_y, mix, standardize, add_intercept, evaluate, evaluate1d
import matplotlib.pyplot as plt

PURCENT = 5 # Purcentage of the set you want on the test set
NUM_FRAMES = 60
# DATADIR = '/baie/corpus/emoMusic/train/'
DATADIR = './train/'

do_regularize = False

y_, song_id, nb_of_songs = load_y(DATADIR)
X_ = load_X(DATADIR, song_id)

# Now  let's mix everything so that we can take test_set and train_set independantly
# We need to separate PER SONG
X_train, y_train, X_test, y_test, song_id_tst = mix(X_, y_, PURCENT, NUM_FRAMES, song_id, nb_of_songs)
print X_train.shape, y_train.shape, X_test.shape, y_test.shape
# print X_train[0:3,0:3]

# standardize data
X_train, scaler = standardize(X_train)
X_test, _ = standardize(X_test, scaler)

# one dimension at a time
y_train = y_train[:,0]
y_test = y_test[:,0]

print X_train.shape, y_train.shape, X_test.shape, y_test.shape

tst_song = len(song_id_tst)

# add column of ones to data to account for the bias:
X_train = add_intercept(X_train)
print X_train.shape
# print X_train[0:10]

data = dict(x=X_train, y=y_train)

with Model() as model:
    # specify glm and pass in data. The resulting linear model, its likelihood and 
    # and all its parameters are automatically added to our model.
    glm.glm('y ~ x', data)
    start = find_MAP()
    step = NUTS(scaling=start) # Instantiate MCMC sampling algorithm
    trace = sample(2000, step, progressbar=False) # draw 2000 posterior samples using NUTS sampling

