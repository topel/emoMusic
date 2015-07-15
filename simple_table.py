__author__ = 'thomas'


# Load modules and data
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# spector_data = sm.datasets.spector.load()
# spector_data.exog = sm.add_constant(spector_data.exog, prepend=False)
#
# # Fit and summarize OLS model
# mod = sm.OLS(spector_data.endog, spector_data.exog)
# res = mod.fit()
# print res.summary()

from utils import load_X, load_y, mix, standardize, add_intercept, evaluate, evaluate1d
import matplotlib.pyplot as plt

PURCENT = 5 # Purcentage of the set you want on the test set
NUM_FRAMES = 60
DATADIR = '/baie/corpus/emoMusic/train/'
# DATADIR = './train/'


EMO='valence'
# EMO='arousal'

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

# select most correlated features
X_train = X_train[:,[10,12,13,17,19,82,83,84,85,89,90,91,103,140,142,146,148,212,214,218,220]]
X_test = X_test[:,[10,12,13,17,19,82,83,84,85,89,90,91,103,140,142,146,148,212,214,218,220]]

# one dimension at a time
# 0: arousal, 1: valence
if EMO == 'valence':
    print '... emotion: valence'
    y_train = y_train[:,0]
    y_test = y_test[:,0]
else:
    print '... emotion: arousal'
    y_train = y_train[:,1]
    y_test = y_test[:,1]

# X_test = X_train[119:119+y_test.shape[0],:]
# y_test = y_train[119:119+y_test.shape[0]]

print X_train.shape, y_train.shape, X_test.shape, y_test.shape

nb_test_song = len(song_id_tst)

dat=np.hstack((y_train[0:10], X_train[0:10,:]))

colnames = ['Y', 'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20']

mystubs = [ 's0', "s1", "s2", 's3', 's4', 's5', 's6', 's7', 's8', 's9' ]
mystubs = range(10)

tbl =  sm.SimpleTable(dat, colnames, mystubs, title="DUMMY")

