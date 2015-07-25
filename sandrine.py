__author__ = 'thomas'

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import cPickle as pickle
import theano
from theano import tensor as T

from utils import load_X, load_y, mix, standardize, add_intercept, evaluate, evaluate1d, load_X_from_fold_to_3dtensor, load_data_from_fold_to_3dtensor_and_genre_info
import matplotlib.pyplot as plt

def write_folds_to_mat_files(normed_folds, num_folds):
    import scipy.io as sio
    for fold in xrange(num_folds):
        print ' ... creating MAT file for fold: %d ...'%(fold)
        train_fold_for_matlab = dict()
        test_fold_for_matlab = dict()
        data = dict()

        for k, val in normed_folds[fold][0].iteritems():
            if (k != 'std' and k != 'mean'):
                new_key = 'song' + k
            else:
                new_key = k
            train_fold_for_matlab[new_key] = val
        data['train'] = train_fold_for_matlab

        for k, val in normed_folds[fold][1].iteritems():
            new_key = 'song' + k
            test_fold_for_matlab[new_key] = val
        data['test'] = test_fold_for_matlab

        # save to MAT
        sio.savemat('train/matfiles/fold%d_normed.mat'%(fold), data, oned_as='row')

def write_fold_to_mat_files(DATADIR, data, fold_id, doNormalize, doDuplicates):
    import scipy.io as sio
    print ' ... creating MAT file for fold: %d ...'%(fold_id)
    # save to MAT
    if doNormalize:
        if doDuplicates:
            nom = DATADIR + 'matfiles/fold%d_normed_3dtensor.mat'%(fold_id)
        else:
            nom = DATADIR + 'matfiles/fold%d_normed_3dtensor_noDuplicates.mat'%(fold_id)
        sio.savemat(nom , data, oned_as='row')
    else:
        if doDuplicates:
            nom = DATADIR + 'matfiles/fold%d_NOT_normed_3dtensor.mat'%(fold_id)
        else:
            nom = DATADIR + 'matfiles/fold%d_NOT_normed_3dtensor_noDuplicates.mat'%(fold_id)
        sio.savemat(nom , data, oned_as='row')
    print '   ---> output: ', nom


NUM_FRAMES = 60
NUM_OUTPUT = 2
NUM_GENRES = 15
DATADIR = '/baie/corpus/emoMusic/train/'
# DATADIR = './train/'
doNormalize = True

# fold_id = 0
for fold_id in range(10):
    print '... loading fold: %d'%(fold_id)
    if doNormalize:
        nom = DATADIR + 'pkl/fold%d_normed.pkl'%(fold_id)
        print 'input file: %s'%(nom)
    else:
        nom = DATADIR + 'pkl/fold%d_NOT_normed.pkl'%(fold_id)
        print 'input file: %s'%(nom)

    fold = pickle.load( open( nom, "rb" ) )
    fold_res = dict()

    for subset in ['train', 'test']:
        print '  ... creating subset: ', subset
        # doDuplicates = True
        # X,  y, song_ids, genre_indexes = load_data_from_fold_to_3dtensor_and_genre_info(fold, subset, NUM_OUTPUT, NUM_GENRES)
        # # print X.shape, y.shape, genre_indexes.shape
        # # (572, 60, 260) (572, 60, 2) (15, 2)
        # # instead of (because of duplicates):
        # # (387, 60, 260) (387, 60, 2) (15, 2)
        # #
        # fold_res[subset] = dict()
        # fold_res[subset]['X'] = X
        # fold_res[subset]['y'] = y
        # fold_res[subset]['genre'] = genre_indexes

        doDuplicates = False
        X,  y, song_ids = load_X_from_fold_to_3dtensor(fold, subset, NUM_OUTPUT)
        # print X.shape, y.shape, song_ids.shape
        # (387, 60, 260) (387, 60, 2) ()
        fold_res[subset] = dict()
        fold_res[subset]['X'] = X
        fold_res[subset]['y'] = y
        fold_res[subset]['song_ids'] = song_ids

    write_fold_to_mat_files(DATADIR, fold_res, fold_id, doNormalize, doDuplicates)


