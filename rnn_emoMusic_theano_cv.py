__author__ = 'thomas'

import rnn_model
# import rnn_model2

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
            fold = pickle.load( open( DATADIR + '/pkl/fold%d_normed_essentia.pkl'%(fold_id), "rb" ) )

        else:
            fold = pickle.load( open( DATADIR + '/pkl/fold%d_normed.pkl'%(fold_id), "rb" ) )
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




def rnn_cv( folds, n_hidden=10, n_epochs=50, lr=0.001, lrd = 0.999, reg_coef= 0.01, doSmoothing=False, useEssentia=False):

    doSaveModel = False

    if doSmoothing:
        dir_name = 'nfeat%d_nh%d_ne%d_lr%g_reg%g_smoothed'%(nb_features, n_hidden, n_epochs, lr, reg_coef)
    else:
        dir_name = 'nfeat%d_nh%d_ne%d_lr%g_reg%g'%(nb_features, n_hidden, n_epochs, lr, reg_coef)
    MODELDIR = 'rnn/' + dir_name + '/'
    LOGDIR = MODELDIR

    if not path.exists(MODELDIR):
        makedirs(MODELDIR)

    print '... output dir: %s'%(MODELDIR)

    # smoothing params
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
    all_fold_id_test = list()

    for fold_id in range(10):
        # fold_id = 0
        fold = folds[fold_id]
        t0 = time.time()

        # print '... loading FOLD %d'%fold_id
        # if useEssentia:
            # fold = pickle.load( open( DATADIR + '/pkl/fold%d_normed_essentia.pkl'%(fold_id), "rb" ) )

        if useEssentia:
            X_train = fold['train']['X']
            y_train = fold['train']['y']
            id_train = fold['train']['song_id']

            X_test = fold['test']['X']
            y_test = fold['test']['y']
            id_test = fold['test']['song_id']

        else:
            fold = pickle.load( open( DATADIR + '/pkl/fold%d_normed.pkl'%(fold_id), "rb" ) )
            X_train, y_train, id_train = load_X_from_fold_to_3dtensor(fold, 'train', NUM_OUTPUT)
            X_test, y_test, id_test = load_X_from_fold_to_3dtensor(fold, 'test', NUM_OUTPUT)


        print X_train.shape, y_train.shape, X_test.shape, y_test.shape

        if useMelodyFeatures:
            # first feature = slope, other two = mean, std
            melody_train, melody_test = subset_features(all_song_melody_features, id_train, id_test)
            # melody_train = melody_train[:,:,1:]
            # melody_test = melody_test[:,:,1:]

            # standardize train data
            melody_concat_train = np.reshape(melody_train, (melody_train.shape[0]*melody_train.shape[1], melody_train.shape[2]), order='C')
            melody_concat_train_normed, scaler = standardize(melody_concat_train)
            # print concat_train_normed.shape
            melody_train_normed = np.reshape(melody_concat_train_normed, (melody_train.shape[0], melody_train.shape[1], melody_train.shape[2]), order='C')
            del melody_concat_train, melody_concat_train_normed

            # standardize test data
            melody_concat_test = np.reshape(melody_test, (melody_test.shape[0]*melody_test.shape[1], melody_test.shape[2]), order='C')
            melody_concat_test_normed, _ = standardize(melody_concat_test, scaler)
            # print concat_test_normed.shape
            melody_test_normed = np.reshape(melody_concat_test_normed, (melody_test.shape[0], melody_test.shape[1], melody_test.shape[2]), order='C')
            del melody_concat_test, melody_concat_test_normed

            # concat with the other features
            X_train = np.concatenate((X_train, melody_train_normed), axis=2)
            X_test = np.concatenate((X_test, melody_test_normed), axis=2)

        if useTempoFeatures:
            tempo_train, tempo_test = subset_features(all_song_tempo_features, id_train, id_test)
            # standardize train data
            tempo_concat_train = np.reshape(tempo_train, (tempo_train.shape[0]*tempo_train.shape[1], tempo_train.shape[2]), order='C')
            tempo_concat_train_normed, scaler = standardize(tempo_concat_train)
            # print concat_train_normed.shape
            tempo_train_normed = np.reshape(tempo_concat_train_normed, (tempo_train.shape[0], tempo_train.shape[1], tempo_train.shape[2]), order='C')
            del tempo_concat_train, tempo_concat_train_normed

            # standardize test data
            tempo_concat_test = np.reshape(tempo_test, (tempo_test.shape[0]*tempo_test.shape[1], tempo_test.shape[2]), order='C')
            tempo_concat_test_normed, _ = standardize(tempo_concat_test, scaler)
            # print concat_test_normed.shape
            tempo_test_normed = np.reshape(tempo_concat_test_normed, (tempo_test.shape[0], tempo_test.shape[1], tempo_test.shape[2]), order='C')
            del tempo_concat_test, tempo_concat_test_normed

            # concat with the other features
            X_train = np.concatenate((X_train, tempo_train_normed), axis=2)
            X_test = np.concatenate((X_test, tempo_test_normed), axis=2)

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
        print y_hat.shape

        if doSmoothing:
            # smooooooth
            y_hat_smooth = np.zeros_like(y_hat, dtype=float)
            for i in xrange(y_hat.shape[0]):
                y_hat_smooth[i, :, 0] = np.convolve(y_hat[i, :, 0], wts, mode='same')
                y_hat_smooth[i, :delay, 0] = y_hat[i, :delay, 0]
                y_hat_smooth[i, -delay:, 0] = y_hat[i, -delay:, 0]
                y_hat_smooth[i, :, 1] = np.convolve(y_hat[i, :, 1], wts, mode='same')
                y_hat_smooth[i, :delay, 1] = y_hat[i, :delay, 1]
                y_hat_smooth[i, -delay:, 1] = y_hat[i, -delay:, 1]


        # save predictions on the test subset, before reshaping to 2-d arrays (I need 3d arrays)
        if doSmoothing:
            # fold_pred = [item for sublist in fold_pred for item in sublist]
            # fold_pred = np.array(fold_pred, dtype=float)
            pred_file = LOGDIR + 'fold%d_test_predictions.pkl'%(fold_id)
            pickle.dump( y_hat_smooth, open( pred_file, "wb" ) )
            print ' ... predictions y_hat_smooth saved in: %s'%(pred_file)
        else:
            # fold_pred = [item for sublist in fold_pred for item in sublist]
            # fold_pred = np.array(fold_pred, dtype=float)
            pred_file = LOGDIR + 'fold%d_test_predictions.pkl'%(fold_id)
            pickle.dump( y_hat, open( pred_file, "wb" ) )
            print ' ... predictions y_hat saved in: %s'%(pred_file)


        if doSmoothing:
            y_hat_smooth = np.reshape(y_hat_smooth, (y_hat_smooth.shape[0]*y_hat_smooth.shape[1], y_hat_smooth.shape[2]))
        y_hat = np.reshape(y_hat, (y_hat.shape[0]*y_hat.shape[1], y_hat.shape[2]))
        y_test_concat = np.reshape(y_test, (y_test.shape[0]*y_test.shape[1], y_test.shape[2]))

        print y_hat.shape, y_test_concat.shape

        assert y_hat.shape == y_test_concat.shape, 'ERROR: pred and ref shapes are different!'

        # concat hyp labels:
        if doSmoothing:
            all_fold_pred.append(y_hat_smooth.tolist())
        else:
            all_fold_pred.append(y_hat.tolist())

        # concat ref labels:
        all_fold_y_test.append(y_test_concat.tolist())

        if doSmoothing:
            RMSE, pcorr, error_per_song, mean_per_song = evaluate(y_test_concat, y_hat_smooth, id_test.shape[0])
        else:
            RMSE, pcorr, error_per_song, mean_per_song = evaluate(y_test_concat, y_hat, id_test.shape[0])

        s = (
                'fold: %d valence: %.4f %.4f arousal: %.4f %.4f\n'
              % (fold_id, RMSE[0], pcorr[0][0], RMSE[1], pcorr[1][0])
        )
        print s
        log_f.write(s)



        # predict on the train set and save predictions (useful to train rnn2)
        if doSmoothing:
            pred = list()
            for ind_seq_train in xrange(nb_seq_train):
                pred.append(model.predict(X_train[ind_seq_train]))

            train_y_hat = np.array(pred, dtype=float)
            print train_y_hat.shape

            train_y_hat_smooth = np.zeros_like(train_y_hat, dtype=float)
            for i in xrange(train_y_hat.shape[0]):
                train_y_hat_smooth[i, :, 0] = np.convolve(train_y_hat[i, :, 0], wts, mode='same')
                train_y_hat_smooth[i, :delay, 0] = train_y_hat[i, :delay, 0]
                train_y_hat_smooth[i, -delay:, 0] = train_y_hat[i, -delay:, 0]
                train_y_hat_smooth[i, :, 1] = np.convolve(train_y_hat[i, :, 1], wts, mode='same')
                train_y_hat_smooth[i, :delay, 1] = train_y_hat[i, :delay, 1]
                train_y_hat_smooth[i, -delay:, 1] = train_y_hat[i, -delay:, 1]

            # no reshape, I need 3d arrays
            # train_y_hat_smooth = np.reshape(train_y_hat_smooth, (train_y_hat_smooth.shape[0]*train_y_hat_smooth.shape[1], train_y_hat_smooth.shape[2]))

            pred_file = LOGDIR + 'fold%d_train_predictions.pkl'%(fold_id)
            pickle.dump( train_y_hat_smooth, open( pred_file, "wb" ) )
            print ' ... predictions y_hat_smooth saved in: %s'%(pred_file)
        else:
            pred = list()
            for ind_seq_train in xrange(nb_seq_train):
                pred.append(model.predict(X_train[ind_seq_train]))

            train_y_hat = np.array(pred, dtype=float)
            pred_file = LOGDIR + 'fold%d_train_predictions.pkl'%(fold_id)
            pickle.dump( train_y_hat, open( pred_file, "wb" ) )
            print ' ... predictions y_hat saved in: %s'%(pred_file)


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

    # save predictions
    pred_file = LOGDIR + 'all_predictions.pkl'
    pickle.dump( all_fold_pred, open( pred_file, "wb" ) )
    print ' ... all predictions saved in: %s'%(pred_file)
    # ref_file = 'rnn/all_groundtruth.pkl'
    # pickle.dump( all_fold_y_test, open( ref_file, "wb" ) )

    # compute t-test p-values with baseline predictions
    baseline_prediction_file = 'rnn/all_baseline_predictions_260feat.pkl'
    baseline_preds = pickle.load(open( baseline_prediction_file, 'r' ))

    pvalue_val = stats.ttest_ind(baseline_preds[:,0], all_fold_pred[:,0])[1]
    pvalue_ar = stats.ttest_ind(baseline_preds[:,1], all_fold_pred[:,1])[1]
    pvalues = (pvalue_val, pvalue_ar)
    RMSE, pcorr, error_per_song, mean_per_song = evaluate(all_fold_y_test, all_fold_pred, 0)

    # print(
    #         'sklearn --> valence: %.4f, arousal: %.4f\n'
    #         'Pearson Corr --> valence: %.4f, arousal: %.4f \n'
    #         # % (RMSE[0], -1. , pcorr[0][0], -1)
    #       % (RMSE[0],RMSE[1],pcorr[0][0], pcorr[1][0])
    # )

    s = (
            'allfolds valence: %.4f %.4f arousal: %.4f %.4f p-values: %.4f, %.4f\n'
          % (RMSE[0], pcorr[0][0], RMSE[1], pcorr[1][0], pvalue_val, pvalue_ar)
    )

    print s
    log_f.write(s)
    log_f.close()
    return RMSE, pcorr, pvalues

if __name__ == '__main__':

    NUM_FRAMES = 60
    NUM_OUTPUT = 2
    DATADIR = '/baie/corpus/emoMusic/train/'
    # DATADIR = './train/'

    useEssentia = False

    if useEssentia:
        # nb_features = 196 # essentia features
        nb_features = 268 # 260 baseline features + 8 essentia features
    else:
        nb_features = 260

    # nb_features += 260

    useMelodyFeatures = False
    if useMelodyFeatures:
        nom = DATADIR + '/pkl/melody_features.pkl'
        all_song_melody_features = pickle.load( open( nom, "rb" ) )
        nb_features += 3

    useTempoFeatures = False
    if useTempoFeatures:
        nom = DATADIR + '/pkl/fixedtempo_features.pkl'
        all_song_tempo_features = pickle.load( open( nom, "rb" ) )
        nb_features += 1

    EMO='valence'
    # EMO='arousal'

    essentia_feat_indices = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 21], [21, 61], [61, 88], [88, 89], [89, 90], [90, 93], [93, 133], [133, 134], [134, 135], [135, 138], [138, 162], [162, 163], [163, 164], [164, 167], [167, 180], [180, 186], [186, 192], [192, 193], [193, 194], [194, 195], [195, 196]]

    baseline_feat_indices = [[0, 2], [2, 4], [4, 6], [6, 8], [8, 10], [10, 12], [12, 14], [14, 16],
    [16, 18], [18, 20], [20, 72], [72, 74], [74, 76], [76, 78], [78, 80], [80, 82], [82, 84],
    [84, 86], [86, 88], [88, 90], [90, 92], [92, 94], [94, 96], [96, 98], [98, 100], [100, 102],
    [102, 130],
    [130, 132], [132, 134], [134, 136], [136, 138], [138, 140], [140, 142], [142, 144], [144, 146], [146, 148], [148, 150],
    [150, 202],
    [202, 204], [204, 206], [206, 208], [208, 210], [210, 212], [212, 214],
    [214, 216], [216, 218], [218, 220], [220, 222], [222, 224], [224, 226], [226, 228], [228, 230], [230, 232],
    [232, 260]]


    # NN params
    n_hidden = 10
    n_epochs = 50
    lr = 0.001
    lrd = 0.999
    reg_coef = 0.01

    # output smoothing with a mean filter
    doSmoothing = True

    print '... useEssentia: %s, useSmoothing: %s'%(useEssentia, doSmoothing)

    # load baseline folds
    folds = load_folds(DATADIR)

    if useEssentia:
        essentia_folds = load_folds(DATADIR, useEssentia)
        ajout_log_file_name = 'ajout_features_flatness_spectralValleys.log'
        ajout_log_file = open(ajout_log_file_name, 'a')

        # for ajout in range(len(essentia_feat_indices)):

        feature_indices_list = list()
        feature_indices_list.append([89, 90])
        feature_indices_list.append([134, 135])
        feature_indices_list.append([186, 192])
        new_folds = add_essentia_features(folds, essentia_folds, feature_indices_list)
        rmse, pcorr, pvalues = rnn_cv(new_folds, n_hidden, n_epochs, lr, lrd, reg_coef, doSmoothing, useEssentia)
        s = (
        'allfolds valence: %.4f %.4f arousal: %.4f %.4f deb:%d, fin:%d\n'
          % (rmse[0], pcorr[0][0], rmse[1], pcorr[1][0], feature_indices_list[0][0], feature_indices_list[0][1])
        )
        print s
        ajout_log_file.write(s)
        ajout_log_file.close()


    else:
        rmse, pcorr, pvalue = rnn_cv(folds, n_hidden, n_epochs, lr, lrd, reg_coef, doSmoothing, useEssentia)

        # retrait_log_file_name = 'retrait_baseline_features.log'
        # retrait_log_file = open(retrait_log_file_name, 'w')
        #
        # # feature_indices_list = [[14, 16], [76, 132], [202, 232]]
        # feature_indices_list = [[84, 130]]
        # # for retrait in range(len(baseline_feat_indices)):
        # #     feature_indices_list = list()
        # #     feature_indices_list.append(baseline_feat_indices[retrait])
        # new_folds = remove_features(folds, feature_indices_list)
        # rmse, pcorr, pvalues = rnn_cv(new_folds, n_hidden, n_epochs, lr, lrd, reg_coef)
        # s = (
        # 'allfolds valence: %.4f %.4f arousal: %.4f %.4f deb:%d, fin:%d p_values: %.3f %.3f\n'
        #   % (rmse[0], pcorr[0][0], rmse[1], pcorr[1][0], feature_indices_list[0][0], feature_indices_list[0][1], pvalues[0], pvalues[1])
        # )
        # print s
        # retrait_log_file.write(s)
        #
        # retrait_log_file.close()

