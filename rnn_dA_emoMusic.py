__author__ = 'tpellegrini'

__author__ = 'thomas'

import rnn_model

import numpy as np
import cPickle as pickle
import logging
import time
import theano
from os import path, makedirs
import matplotlib.pyplot as plt
from scipy import stats

from utils import evaluate, load_X_from_fold_to_3dtensor, subset_features, standardize

import sys
# import settings
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO,filename='example.log')


mode = theano.Mode(linker='cvm')
#mode = 'DEBUG_MODE'

def load_folds(data_dir, useEssentia=False):
    folds = list()
    for fold_id in range(10):
        print '... loading FOLD %d'%fold_id
        if useEssentia:
            fold = pickle.load( open( data_dir + '/pkl/fold%d_normed_essentia.pkl'%(fold_id), "rb" ) )

        else:
            fold = pickle.load( open( data_dir + '/pkl/fold%d_normed.pkl'%(fold_id), "rb" ) )
        folds.append(fold)

    return folds

def load_prediction_folds(data_dir):
    folds = list()
    for fold_id in range(10):
        print '... loading FOLD %d'%fold_id
        fold = pickle.load( open( data_dir + '/fold%d_predictions.pkl'%(fold_id), "rb" ) )
        folds.append(fold)

    return folds


def add_essentia_features(folds, essentia_folds, feature_indices_list):
    new_folds = list()
    for fold_id in range(10):
        # fold_id = 0
        fold = folds[fold_id]
        X_train, y_train, id_train = load_X_from_fold_to_3dtensor(fold, 'train', NUM_OUTPUT)
        X_test, y_test, id_test = load_X_from_fold_to_3dtensor(fold, 'test', NUM_OUTPUT)

        essentia_fold = essentia_folds[fold_id]
        essentia_X_train = essentia_fold['train']['X']
        essentia_X_test = essentia_fold['test']['X']

        for seg in feature_indices_list:
            deb = seg[0]
            fin = seg[1]
            X_train = np.concatenate((X_train, essentia_X_train[:,:,deb:fin]), axis=2)
            X_test = np.concatenate((X_test, essentia_X_test[:,:,deb:fin]), axis=2)

        print X_train.shape
        data = dict()
        data['train'] = dict()
        data['train']['X'] = X_train
        data['train']['y'] = y_train
        data['train']['song_id'] = id_train
        data['test'] = dict()
        data['test']['X'] = X_test
        data['test']['y'] = y_test
        data['test']['song_id'] = id_test

        new_folds.append(data)
    return new_folds

def add_prediction_features(folds, train_pred_folds, test_pred_folds, feature_indices_list):
    new_folds = list()
    for fold_id in range(10):
        # fold_id = 0
        fold = folds[fold_id]
        X_train, y_train, id_train = load_X_from_fold_to_3dtensor(fold, 'train', NUM_OUTPUT)
        X_test, y_test, id_test = load_X_from_fold_to_3dtensor(fold, 'test', NUM_OUTPUT)

        train_pred_fold = train_pred_folds[fold_id]
        test_pred_fold = test_pred_folds[fold_id]

        for seg in feature_indices_list:
            deb = seg[0]
            fin = seg[1]
            X_train = np.concatenate((X_train, train_pred_fold[:,:,deb:fin]), axis=2)
            X_test = np.concatenate((X_test, test_pred_fold[:,:,deb:fin]), axis=2)

        print X_train.shape
        data = dict()
        data['train'] = dict()
        data['train']['X'] = X_train
        data['train']['y'] = y_train
        data['train']['song_id'] = id_train
        data['test'] = dict()
        data['test']['X'] = X_test
        data['test']['y'] = y_test
        data['test']['song_id'] = id_test

        new_folds.append(data)
    return new_folds

def remove_features(folds, feature_indices_list):
    new_folds = list()

    bool_feature_mask = np.ones(260)
    for seg in feature_indices_list:
        deb = seg[0]
        fin = seg[1]
        bool_feature_mask[deb:fin] = 0
    bool_feature_mask = bool_feature_mask == 1

    for fold_id in range(10):
        # fold_id = 0
        fold = folds[fold_id]
        X_train, y_train, id_train = load_X_from_fold_to_3dtensor(fold, 'train', NUM_OUTPUT)
        X_test, y_test, id_test = load_X_from_fold_to_3dtensor(fold, 'test', NUM_OUTPUT)

        X_train = X_train[:,:,bool_feature_mask]
        X_test = X_test[:,:,bool_feature_mask]

        print X_train.shape, X_test.shape

        data = dict()
        data['train'] = dict()
        data['train']['X'] = X_train
        data['train']['y'] = y_train
        data['train']['song_id'] = id_train
        data['test'] = dict()
        data['test']['X'] = X_test
        data['test']['y'] = y_test
        data['test']['song_id'] = id_test

        new_folds.append(data)
    return new_folds

def rnn_main( fold, n_hidden=10, n_epochs=100, lr=0.001, lrd = 0.999, reg_coef= 0.01):

    doSaveModel = False
    MODELDIR = 'AE/models/'
    LOGDIR = MODELDIR

    if not path.exists(MODELDIR):
        makedirs(MODELDIR)

    print '... output dir: %s'%(MODELDIR)

    # # initialize global logger variable
    # print '... initializing global logger variable'
    # logger = logging.getLogger(__name__)
    # withFile = False
    # logger = settings.init(MODELDIR + 'train.log', withFile)

    # perf_file_name = LOGDIR + 'rnn_nh%d_ne%d_lr%g_reg%g.log'%(n_hidden, n_epochs, lr, reg_coef)
    perf_file_name = LOGDIR + 'performance.log'
    log_f = open(perf_file_name, 'w')

    all_fold_pred = list()
    all_fold_y_test = list()
    all_fold_id_test = list()

    # for fold_id in range(10):
        # fold_id = 0
    # fold = folds[fold_id]
    t0 = time.time()

    # print '... loading FOLD %d'%fold_id
    # if useEssentia:
        # fold = pickle.load( open( DATADIR + '/pkl/fold%d_normed_essentia.pkl'%(fold_id), "rb" ) )
    X_train = fold['train']['X']
    y_train = fold['train']['y']
    id_train = fold['train']['song_id']

    X_test = fold['test']['X']
    y_test = fold['test']['y']
    id_test = fold['test']['song_id']

    X_train = np.reshape(X_train, (X_train.shape[0]//60, 60, X_train.shape[1]), order='C')
    X_test = np.reshape(X_test, (X_test.shape[0]//60, 60, X_test.shape[1]), order='C')
    y_train = np.reshape(y_train, (y_train.shape[0]//60, 60, y_train.shape[1]), order='C')
    y_test = np.reshape(y_test, (y_test.shape[0]//60, 60, y_test.shape[1]), order='C')

    # else:
    #     # fold = pickle.load( open( DATADIR + '/pkl/fold%d_normed.pkl'%(fold_id), "rb" ) )
    #     X_train, y_train, id_train = load_X_from_fold_to_3dtensor(fold, 'train', NUM_OUTPUT)
    #     X_test, y_test, id_test = load_X_from_fold_to_3dtensor(fold, 'test', NUM_OUTPUT)

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

    model = rnn_model.MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                    learning_rate=lr, learning_rate_decay=lrd,
                    L1_reg=reg_coef, L2_reg=reg_coef,
                    n_epochs=n_epochs, activation='tanh')

    model.fit(X_train, y_train, validation_frequency=validation_frequency)

    if doSaveModel:
        # model_name = MODELDIR + 'rnn_fold%d_nh%d_nepochs%d_lr%g_reg%g.pkl'%(fold_id, n_hidden, n_epochs, lr, reg_coef)
        model_name = MODELDIR + 'model_fold%d.pkl'%(fold_id)
        model.save(fpath=model_name)

    pred = list()
    for ind_seq_test in xrange(nb_seq_test):
        pred.append(model.predict(X_test[ind_seq_test]))

    y_hat = np.array(pred, dtype=float)
    y_hat = np.reshape(y_hat, (y_hat.shape[0]*y_hat.shape[1], y_hat.shape[2]))

    y_test_concat = np.reshape(y_test, (y_test.shape[0]*y_test.shape[1], y_test.shape[2]))

    print y_hat.shape, y_test_concat.shape

    assert y_hat.shape == y_test_concat.shape, 'ERROR: pred and ref shapes are different!'

    # all_fold_pred.append(y_hat.tolist())
    # all_fold_y_test.append(y_test_concat.tolist())

    RMSE, pcorr, error_per_song, mean_per_song = evaluate(y_test_concat, y_hat, id_test.shape[0])

    s = (
            'fold: %d valence: %.4f %.4f arousal: %.4f %.4f\n'
          % (fold_id, RMSE[0], pcorr[0][0], RMSE[1], pcorr[1][0])
    )
    print s
    log_f.write(s)

    fold_prediction_file = 'AE/pred/fold%d.pkl'%(fold_id)
    pickle.dump( y_test_concat, open( fold_prediction_file, "wb" ) )

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

    # all_fold_pred = [item for sublist in all_fold_pred for item in sublist]
    # all_fold_y_test = [item for sublist in all_fold_y_test for item in sublist]
    #
    # all_fold_pred = np.array(all_fold_pred, dtype=float)
    # all_fold_y_test = np.array(all_fold_y_test, dtype=float)
    #
    # print all_fold_pred.shape, all_fold_y_test.shape
    #
    # # save predictions
    # pred_file = LOGDIR + 'all_predictions.pkl'
    # pickle.dump( all_fold_pred, open( pred_file, "wb" ) )
    # print ' ... all predictions saved in: %s'%(pred_file)
    # # ref_file = 'rnn/all_groundtruth.pkl'
    # # pickle.dump( all_fold_y_test, open( ref_file, "wb" ) )
    #
    # # compute t-test p-values with baseline predictions
    # baseline_prediction_file = 'rnn/all_baseline_predictions_260feat.pkl'
    # baseline_preds = pickle.load(open( baseline_prediction_file, 'r' ))
    #
    # pvalue_val = stats.ttest_ind(baseline_preds[:,0], all_fold_pred[:,0])[1]
    # pvalue_ar = stats.ttest_ind(baseline_preds[:,1], all_fold_pred[:,1])[1]
    # pvalues = (pvalue_val, pvalue_ar)
    # RMSE, pcorr, error_per_song, mean_per_song = evaluate(all_fold_y_test, all_fold_pred, 0)
    #
    # # print(
    # #         'sklearn --> valence: %.4f, arousal: %.4f\n'
    # #         'Pearson Corr --> valence: %.4f, arousal: %.4f \n'
    # #         # % (RMSE[0], -1. , pcorr[0][0], -1)
    # #       % (RMSE[0],RMSE[1],pcorr[0][0], pcorr[1][0])
    # # )
    #
    # s = (
    #         'allfolds valence: %.4f %.4f arousal: %.4f %.4f p-values: %.4f, %.4f\n'
    #       % (RMSE[0], pcorr[0][0], RMSE[1], pcorr[1][0], pvalue_val, pvalue_ar)
    # )
    #
    # print s
    # log_f.write(s)
    log_f.close()
    # return RMSE, pcorr, pvalues
    return RMSE

if __name__ == '__main__':

    nb_features = 500

    print '... training with %d features ...'%(nb_features)
    cost_type = 'MSE'
    noise_type = 'gaussian'
    corruption_level=0.3
    n_hidden=500
    training_epochs=100
    fold_id = 0

    act_dir = 'AE/activations/'
    MODELDIR = 'AE/models/da_rnn_fold%d_cost%s_noise%s_level%.1f_nh%d_it%d.pkl'%(fold_id, cost_type, noise_type, corruption_level, n_hidden, training_epochs)
    model_file = 'model.pkl'
    predictions = MODELDIR + 'predictions.pkl'

    if not path.exists(MODELDIR):
        makedirs(MODELDIR)

    folds = list()
    for fold_id in range(10):
        print 'fold_id: %d'%(fold_id)
        data_file = act_dir + 'fold%d_cost%s_noise%s_level%.1f_nh%d_it%d.pkl'%(fold_id, cost_type, noise_type, corruption_level, n_hidden, training_epochs)
        data = pickle.load( open( data_file, "rb" ) )
    #    folds.append(data)

        RMSE = rnn_main( data, n_hidden=500, n_epochs=100, lr=0.001, lrd = 0.999, reg_coef= 0.01)


