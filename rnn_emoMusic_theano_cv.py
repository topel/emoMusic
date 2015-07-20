__author__ = 'thomas'

import rnn_model

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import cPickle as pickle
import datetime
import logging
import time
import theano

from utils import load_X, load_y, mix, standardize, add_intercept, evaluate, evaluate1d, load_X_from_fold_to_3dtensor
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO,filename='example.log')

mode = theano.Mode(linker='cvm')
#mode = 'DEBUG_MODE'

NUM_FRAMES = 60
NUM_OUTPUT = 2
DATADIR = '/baie/corpus/emoMusic/train/'
# DATADIR = './train/'
MODELDIR = 'models/'
LOGDIR = 'log/'

EMO='valence'
# EMO='arousal'

# fold_id = 2

all_fold_pred = list()
all_fold_y_test = list()
all_fold_id_test = list()

n_hidden = 10
n_epochs = 50
lr = 0.001
reg_coef = 0.01

log_file_name = LOGDIR + 'rnn_nh%d_ne%d_lr%g_reg%g.log'%(n_hidden, n_epochs, lr, reg_coef)
log_f = open(log_file_name, 'w')

for fold_id in range(10):
    t0 = time.time()

    print '... loading FOLD %d'%fold_id
    fold = pickle.load( open( DATADIR + '/pkl/fold%d_normed.pkl'%(fold_id), "rb" ) )

    X_train, y_train, id_train = load_X_from_fold_to_3dtensor(fold, 'train', NUM_OUTPUT)
    X_test, y_test, id_test = load_X_from_fold_to_3dtensor(fold, 'test', NUM_OUTPUT)
    # print id_test.shape

    # X_train = X_train[0:100,:,:]
    # y_train = y_train[0:100,:,:]

    # X_train = X_train[:,[10,12,13,17,19,82,83,84,85,89,90,91,103,140,142,146,148,212,214,218,220]]
    # X_test = X_test[:,[10,12,13,17,19,82,83,84,85,89,90,91,103,140,142,146,148,212,214,218,220]]
    # X_train = X_train[:,[13,85,103,142,214]]
    # X_test = X_test[:,[13,85,103,142,214]]

    # X_test = X_train[119:119+y_test.shape[0],:]
    # y_test = y_train[119:119+y_test.shape[0]]

    print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    nb_seq_train, nb_frames_train, nb_features_train = X_train.shape
    nb_seq_test, nb_frames_test, nb_features_test = X_test.shape

    assert nb_frames_train == nb_frames_test, 'ERROR: nb of frames differ from TRAIN to TEST'
    assert nb_features_train == nb_features_test, 'ERROR: nb of features differ from TRAIN to TEST'

    dim_ouput_train = y_train.shape[2]
    dim_ouput_test = y_test.shape[2]

    assert dim_ouput_test == dim_ouput_train, 'ERROR: nb of targets differ from TRAIN to TEST'


    n_in = nb_features_train
    n_out = dim_ouput_train
    n_steps = nb_frames_train

    validation_frequency = nb_seq_train * 2 # for logging during training: every 2 epochs
    # np.random.seed(0)
    # # simple lag test
    # seq = np.random.randn(nb_seq, n_steps, n_in)
    # targets = np.zeros((nb_seq, n_steps, n_out))
    #
    # targets[:, 1:, 0] = seq[:, :-1, 3]  # delayed 1
#    targets[:, 1:, 1] = seq[:, :-1, 2]  # delayed 1
#    targets[:, 2:, 2] = seq[:, :-2, 0]  # delayed 2

    # targets += 0.01 * np.random.standard_normal(targets.shape)

    model = rnn_model.MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                    learning_rate=lr, learning_rate_decay=0.999,
                    L1_reg=reg_coef, L2_reg=reg_coef,
                    n_epochs=n_epochs, activation='tanh')

    model.fit(X_train, y_train, validation_frequency=validation_frequency)

    model_name = MODELDIR + 'rnn_fold%d_nh%d_nepochs%d_lr%g_reg%g.pkl'%(fold_id, n_hidden, n_epochs, lr, reg_coef)
    model.save(fpath=model_name)

    pred = list()
    for ind_seq_test in xrange(nb_seq_test):
        pred.append(model.predict(X_test[ind_seq_test]))

    y_hat = np.array(pred, dtype=float)
    y_hat = np.reshape(y_hat, (y_hat.shape[0]*y_hat.shape[1], y_hat.shape[2]))

    y_test_concat = np.reshape(y_test, (y_test.shape[0]*y_test.shape[1], y_test.shape[2]))

    print y_hat.shape, y_test_concat.shape

    assert y_hat.shape == y_test_concat.shape, 'ERROR: pred and ref shapes are different!'

    all_fold_pred.append(y_hat.tolist())
    all_fold_y_test.append(y_test_concat.tolist())

    RMSE, pcorr, error_per_song, mean_per_song = evaluate(y_test_concat, y_hat, id_test.shape[0])

    s = (
            'fold: %d valence: %.4f %.4f arousal: %.4f %.4f\n'
          % (fold_id, RMSE[0], pcorr[0][0], RMSE[1], pcorr[1][0])
    )
    print s
    log_f.write(s)

    doPlot = False
    if doPlot:
        fig, ax = plt.subplots()
        x1 = np.linspace(1, y_test_concat.shape[0], y_test_concat.shape[0])
        if EMO == 'valence':
            ax.plot(x1, y_test_concat[:, 0], 'o', label="Data")
            # ax.plot(x1, y_hat[:,0], 'r-', label="OLS prediction")
            ax.plot(x1, y_hat[:,0], 'ro', label="OLS prediction")
        else:
            ax.plot(x1, y_test_concat[:, 1], 'o', label="Data")
            ax.plot(x1, y_hat[:,1], 'ro', label="OLS prediction")

        plt.title(EMO + ' on Test subset')
        ax.legend(loc="best")
        plt.show()
        # plt.savefig('figures/rnn_%s_fold%d.png'%(EMO, fold_id), format='png')


    doPlotTrain = False
    if doPlotTrain:
        # plt.close('all')
        fig = plt.figure()
        ax1 = plt.subplot(211)
        plt.plot(X_train[0])
        ax1.set_title('input')

        ax2 = plt.subplot(212)
        true_targets = plt.plot(y_train[0])

        guess = model.predict(X_train[0])
        guessed_targets = plt.plot(guess, linestyle='--')
        for i, x in enumerate(guessed_targets):
            x.set_color(true_targets[i].get_color())
        ax2.set_title('solid: true output, dashed: model output')
        plt.show()

    doPlotTest = False
    if doPlotTest:
        # plt.close('all')
        fig = plt.figure()
        ax1 = plt.subplot(211)
        plt.plot(X_test[0])
        ax1.set_title('input')

        ax2 = plt.subplot(212)
        true_targets = plt.plot(y_test[0])

        # guess = model.predict(X_test[0])
        guess = y_hat[0]

        guessed_targets = plt.plot(guess, linestyle='--')
        for i, x in enumerate(guessed_targets):
            x.set_color(true_targets[i].get_color())
        ax2.set_title('solid: true output, dashed: model output')
        plt.show()

    print "... Elapsed time: %f" % (time.time() - t0)

all_fold_pred = [item for sublist in all_fold_pred for item in sublist]
all_fold_y_test = [item for sublist in all_fold_y_test for item in sublist]

all_fold_pred = np.array(all_fold_pred, dtype=float)
all_fold_y_test = np.array(all_fold_y_test, dtype=float)

print all_fold_pred.shape, all_fold_y_test.shape

RMSE, pcorr, error_per_song, mean_per_song = evaluate(all_fold_y_test, all_fold_pred, 0)

# print(
#         'sklearn --> valence: %.4f, arousal: %.4f\n'
#         'Pearson Corr --> valence: %.4f, arousal: %.4f \n'
#         # % (RMSE[0], -1. , pcorr[0][0], -1)
#       % (RMSE[0],RMSE[1],pcorr[0][0], pcorr[1][0])
# )

s = (
        'allfolds valence: %.4f %.4f arousal: %.4f %.4f\n'
      % (RMSE[0], pcorr[0][0], RMSE[1], pcorr[1][0])
)

print s
log_f.write(s)
log_f.close()