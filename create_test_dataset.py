__author__ = 'thomas'

import numpy as np
from utils import load_TEST_data_to_song_dict, create_folds, standardize_folds, standardize, add_intercept
import matplotlib.pyplot as plt
from sklearn import preprocessing
import cPickle as pickle

if __name__ == '__main__':
    NUM_FRAMES = 60
    NUM_OUTPUT = 2
    DATADIR = '/baie/corpus/emoMusic/test/'
    # DATADIR = './test/'
    doNormalize = False

    metadatafile = DATADIR + 'metadata_testset_2015.csv'

    song_data_dict = load_TEST_data_to_song_dict(metadatafile, DATADIR)

    print '... create a single 3d tensor ...'
    # pick an item to get dimensions
    tmp = song_data_dict.itervalues().next()
    frame_dim, feature_dim = tmp['X'].shape
    nb_of_items = len(song_data_dict)
    X_ = np.zeros((nb_of_items, frame_dim, feature_dim), dtype = float)

    song_ids = list()
    ind_sequence = 0
    for k, v in song_data_dict.iteritems():
        if (k == 'std' or k == 'mean'):
            continue
        song_ids.append(k)
        X_[ind_sequence] = v['X']
        ind_sequence += 1

    song_ids = np.array(song_ids, dtype=int)

    # print '... saving to PKL file ... '
    # data = dict()
    # data['test'] = dict()
    # data['test']['X'] = X_
    # data['test']['song_id'] = song_ids
    # nom = DATADIR + 'pkl/test_set_baseline_260features_58songs_NOT_normed.pkl'
    # pickle.dump( data, open( nom, "wb" ) )
    # print ' ... --> saved to: ** %s **'%(nom)

    print '\n... standardizing data ... '

    X_2d = np.reshape(X_, (X_.shape[0]*X_.shape[1], X_.shape[2]), order='C')

    scaler = preprocessing.StandardScaler()

    # load means and variances from the training dataset
    train_file = '/baie/corpus/emoMusic/train/pkl/train_set_baseline_260features_431songs_normed.pkl'
    train_data = pickle.load( open( train_file, "rb" ) )

    means = train_data['train']['mean']
    stds = train_data['train']['std']

    scaler.mean_ = means
    scaler.std_ = stds

    X_2d_normed = scaler.transform(X_2d)

    X_normed = np.reshape(X_2d_normed, (X_.shape[0], X_.shape[1], X_.shape[2]), order='C')

    print 'before: '
    print np.mean(X_2d, axis=0)
    print np.std(X_2d, axis=0)
    print 'after: '
    print np.mean(X_2d_normed, axis=0)
    print np.std(X_2d_normed, axis=0)

    data = dict()
    data['test'] = dict()
    data['test']['X'] = X_normed
    data['test']['song_id'] = song_ids
    nom = DATADIR + '/pkl/test_set_baseline_260features_58songs_normed.pkl'
    pickle.dump( data, open( nom, "wb" ) )
    print ' ... --> saved to: ** %s **'%(nom)

