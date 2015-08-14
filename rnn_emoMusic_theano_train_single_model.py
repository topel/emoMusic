__author__ = 'thomas'

import rnn_model
import rnn_model2

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

def remove_features(folds, feature_indices_list, NUM_OUTPUT):
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


def rnn_cv( output_model_dir, model_name, pred_file, data, n_hidden=10, n_epochs=50, lr=0.001, lrd = 0.999, reg_coef= 0.01, doSmoothing=False):

    doSaveModel = True

    MODELDIR = output_model_dir
    LOGDIR = MODELDIR
    print '... model output dir: %s'%(MODELDIR)

    # smooth prediction params
    taille = 12
    wts = np.ones(taille-1)*1./taille
    wts = np.hstack((np.array([1./(2*taille)]), wts, np.array([1./(2*taille)])))
    delay = (wts.shape[0]-1) / 2

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
    all_fold_pred_smooth = list()

    t0 = time.time()

    X_train = data['train']['X']
    y_train = data['train']['y']
    id_train = data['train']['song_id']

    X_test = X_train
    y_test = y_train
    id_test = id_train

    print '... training and testing on the same data ...'
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
        # model_name = MODELDIR + 'model_baseline_predictions_as_features_431songs_normed.pkl'
        model_name = MODELDIR + model_name
        model.save(fpath=model_name)

    pred = list()
    pred_smooth = list()

    for ind_seq_test in xrange(nb_seq_test):
        pred.append(model.predict(X_test[ind_seq_test]))

    if doSmoothing:
        for ind_seq_test in xrange(nb_seq_test):
            y_hat = np.array(model.predict(X_test[ind_seq_test]), dtype=float)
            y_hat_smooth = np.zeros_like(y_hat, dtype=float)
            y_hat_smooth[:, 0] = np.convolve(y_hat[:, 0], wts, mode='same')
            y_hat_smooth[:delay, 0] = y_hat[:delay, 0]
            y_hat_smooth[-delay:, 0] = y_hat[-delay:, 0]
            y_hat_smooth[:, 1] = np.convolve(y_hat[:, 1], wts, mode='same')
            y_hat_smooth[:delay, 1] = y_hat[:delay, 1]
            y_hat_smooth[-delay:, 1] = y_hat[-delay:, 1]
            pred_smooth.append(y_hat_smooth)

    y_hat = np.array(pred, dtype=float)
    y_hat_smooth = np.array(pred_smooth, dtype=float)

    # save predictions as 3d tensors
    # pred_file = LOGDIR + 'predictions_train_set_baseline_predictions_as_features_431songs_normed.pkl'
    if doSmoothing:
        pickle.dump( y_hat_smooth, open( pred_file, "wb" ) )
    else:
        pickle.dump( y_hat, open( pred_file, "wb" ) )
    print ' ... predictions saved in: %s'%(pred_file)

    y_hat = np.reshape(y_hat, (y_hat.shape[0]*y_hat.shape[1], y_hat.shape[2]))
    y_test_concat = np.reshape(y_test, (y_test.shape[0]*y_test.shape[1], y_test.shape[2]))

    print y_hat.shape, y_test_concat.shape

    assert y_hat.shape == y_test_concat.shape, 'ERROR: pred and ref shapes are different!'

    all_fold_pred.append(y_hat.tolist())
    all_fold_y_test.append(y_test_concat.tolist())

    RMSE, pcorr, error_per_song, mean_per_song = evaluate(y_test_concat, y_hat, id_test.shape[0])

    s = (
            'training_data valence: %.4f %.4f arousal: %.4f %.4f\n'
          % (RMSE[0], pcorr[0][0], RMSE[1], pcorr[1][0])
    )
    print s
    log_f.write(s)

    if doSmoothing:
        y_hat_smooth = np.reshape(y_hat_smooth, (y_hat_smooth.shape[0]*y_hat_smooth.shape[1], y_hat_smooth.shape[2]))
    # all_fold_pred_smooth.append(y_hat_smooth.tolist())

        RMSE, pcorr, error_per_song, mean_per_song = evaluate(y_test_concat, y_hat_smooth, id_test.shape[0])

        s = (
                'training_data valence: %.4f %.4f arousal: %.4f %.4f\n'
              % (RMSE[0], pcorr[0][0], RMSE[1], pcorr[1][0])
        )
        print s
        log_f.write(s)

    EMO = 'valence'
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

        plt.title(EMO + ' on Train subset')
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

    return RMSE, pcorr

if __name__ == '__main__':

    doUseEssentiaFeatures = True
    doTrainFirstRNN = True
    doTrainSecondRNN = False
    doSmoothing=False # remark: True only for rnn1 and False for rnn2

    if doUseEssentiaFeatures:
        nb_features = 268
    else:
        nb_features = 260

    print '... training with %d features ...'%(nb_features)
    if doTrainFirstRNN:
        train_file = '/baie/corpus/emoMusic/train/pkl/train_set_baseline_%dfeatures_431songs_normed.pkl'%(nb_features)
        MODELDIR = 'RNN_models/rnn1_baseline_%dfeat_nh10_ne50_lr0.001_reg0.01/'%(nb_features)
        model_file = 'model_baseline_%dfeatures_431songs_normed.pkl'%(nb_features)
        if doSmoothing:
            predictions = MODELDIR + 'smoothed_predictions_train_set_baseline_%dfeatures_431songs_normed.pkl'%(nb_features)
        else:
            predictions = MODELDIR + 'predictions_train_set_baseline_%dfeatures_431songs_normed.pkl'%(nb_features)

        if not path.exists(MODELDIR):
            makedirs(MODELDIR)
        train_data = pickle.load( open( train_file, "rb" ) )

        RMSE, pcorr = rnn_cv(MODELDIR, model_file, predictions, train_data, doSmoothing=doSmoothing)

    if doTrainSecondRNN:
        # train a model with the predictions as features
        train_file1 = '/baie/corpus/emoMusic/train/pkl/train_set_baseline_%dfeatures_431songs_normed.pkl'%(nb_features)
        MODELDIR1 = 'RNN_models/rnn1_baseline_%dfeat_nh10_ne50_lr0.001_reg0.01/'%(nb_features)
        MODELDIR2 = 'RNN_models/rnn2_predictions_as_features_rnn1_baseline_%dfeat_nh10_ne50_lr0.001_reg0.01/'%(nb_features)

        if doSmoothing:
            predictions1 = MODELDIR1 + 'smoothed_predictions_train_set_baseline_%dfeatures_431songs_normed.pkl'%(nb_features)
            model_file = 'smoothed_model_baseline_predictions_as_features_431songs_normed.pkl'
            predictions2 = MODELDIR2 + 'smoothed_predictions_train_set_baseline_%dfeatures_431songs_normed.pkl'%(nb_features)
        else:
            predictions1 = MODELDIR1 + 'predictions_train_set_baseline_%dfeatures_431songs_normed.pkl'%(nb_features)
            model_file = 'model_baseline_predictions_as_features_431songs_normed.pkl'
            predictions2 = MODELDIR2 + 'predictions_train_set_baseline_%dfeatures_431songs_normed.pkl'%(nb_features)


        if not path.exists(MODELDIR2):
            makedirs(MODELDIR2)

        train_data1 = pickle.load( open( train_file1, "rb" ) )
        print '... loading train set predictions ...'
        data = pickle.load( open( predictions1, 'rb' ) )
        train_data2 = dict()
        train_data2['train'] = dict()
        train_data2['train']['X'] = data
        train_data2['train']['y'] = train_data1['train']['y']
        train_data2['train']['song_id'] = train_data1['train']['song_id']

        RMSE, pcorr = rnn_cv( MODELDIR2, model_file, predictions2, train_data2, doSmoothing=False )
