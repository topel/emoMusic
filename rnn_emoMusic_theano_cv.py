__author__ = 'thomas'

import rnn_model

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import cPickle as pickle
import theano
from theano import tensor as T

from utils import load_X, load_y, mix, standardize, add_intercept, evaluate, evaluate1d, load_X_from_fold_to_3dtensor
import matplotlib.pyplot as plt



PURCENT = 5 # Purcentage of the set you want on the test set
NUM_FRAMES = 60
# DATADIR = '/baie/corpus/emoMusic/train/'
DATADIR = './train/'


EMO='valence'
# EMO='arousal'
do_regularize = False

# fold_id = 2

all_fold_pred = list()
all_fold_y_test = list()
all_fold_id_test = list()

for fold_id in range(1):
    print '... loading FOLD %d'%fold_id
    fold = pickle.load( open( DATADIR + '/pkl/fold%d_normed.pkl'%(fold_id), "rb" ) )

    X_train, y_train, id_train = load_X_from_fold_to_3dtensor(fold, 'train')
    X_test, y_test, id_test = load_X_from_fold_to_3dtensor(fold, 'test')

    print id_test.shape

    # X_train = X_train[:,[10,12,13,17,19,82,83,84,85,89,90,91,103,140,142,146,148,212,214,218,220]]
    # X_test = X_test[:,[10,12,13,17,19,82,83,84,85,89,90,91,103,140,142,146,148,212,214,218,220]]
    # X_train = X_train[:,[13,85,103,142,214]]
    # X_test = X_test[:,[13,85,103,142,214]]

    # one dimension at a time
    # 0: arousal, 1: valence
    if EMO == 'valence':
        print '... emotion: valence'
        y_train = y_train[0:100,0]
        y_test = y_test[0:100,0]
    else:
        print '... emotion: arousal'
        y_train = y_train[:,1]
        y_test = y_test[:,1]

    # X_test = X_train[119:119+y_test.shape[0],:]
    # y_test = y_train[119:119+y_test.shape[0]]

    print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    nb_features = X_train.shape[1]

    n_hidden = 10
    n_in = nb_features
    n_out = 1 # try 2
    n_steps = NUM_FRAMES
    n_seq = 100

    np.random.seed(0)
    # simple lag test
    seq = np.random.randn(n_seq, n_steps, n_in)
    targets = np.zeros((n_seq, n_steps, n_out))

    targets[:, 1:, 0] = seq[:, :-1, 3]  # delayed 1
#    targets[:, 1:, 1] = seq[:, :-1, 2]  # delayed 1
#    targets[:, 2:, 2] = seq[:, :-2, 0]  # delayed 2

    targets += 0.01 * np.random.standard_normal(targets.shape)

    model = rnn_model.MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                    learning_rate=0.001, learning_rate_decay=0.999,
                    n_epochs=400, activation='tanh')

    model.fit(seq, targets, validation_frequency=1000)

    plt.close('all')
    fig = plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(seq[0])
    ax1.set_title('input')

    ax2 = plt.subplot(212)
    true_targets = plt.plot(targets[0])

    guess = model.predict(seq[0])
    guessed_targets = plt.plot(guess, linestyle='--')
    for i, x in enumerate(guessed_targets):
        x.set_color(true_targets[i].get_color())
    ax2.set_title('solid: true output, dashed: model output')
    plt.show()

