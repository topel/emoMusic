# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:49:37 2015

@author: barriere
"""

import numpy as np
from sklearn import linear_model
from utils import load_X, load_y, mix, evaluate
# import matplotlib.pyplot as plt

PURCENT = 5 # Purcentage of the set you want on the test set
NUM_FRAMES = 60
# DATADIR = '/baie/corpus/emoMusic/train/'
DATADIR = 'train/'
y, song_id, nb_of_songs = load_y(DATADIR)
X = load_X(DATADIR, song_id)

# Now  let's mix everything so that we can take test_set and train_set independantly
# We need to separate PER SONG

X_train, y_train, X_test, y_test, song_id_tst = mix(X, y, PURCENT, NUM_FRAMES, song_id, nb_of_songs)
print X_train.shape, y_train.shape, X_test.shape, y_test.shape

tst_song = len(song_id_tst)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# The coefficients
# print('Coefficients: \n', regr.coef_)

# What we predict
y_hat = regr.predict(X_test)

RMSE, pcorr, error_per_song, mean_per_song = evaluate(y_test, y_hat, tst_song)

All_stack =np.hstack(( error_per_song, mean_per_song ))
print'  Error per song (ar/val)  Mean_per_song (ar/val)    :\n'
print(All_stack)
print '\n'

print'song_id :'
print(song_id_tst)
print '\n'
#print('Error per song: \n', Error_per_song)

print(
        'sklearn --> arrousal : %.4f, valence : %.4f\n'
        'Pearson Corr --> arrousal : %.4f, valence : %.4f \n'
      % (RMSE[0],RMSE[1],pcorr[0][0], pcorr[1][0])
)

# Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % regr.score(X_test, y_test))

# print 'Regression on %s with %d purcent of the datas on testset' %(MODE,PURCENT)
## Plot outputs
#plt.scatter(X_test, y_test,  color='black')
#plt.plot(X_test, regr.predict(X_test), color='blue',
#         linewidth=3)
#
#plt.xticks(())
#plt.yticks(())
#
#plt.show()
