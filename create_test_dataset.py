__author__ = 'thomas'

import numpy as np
from utils import load_TEST_data_to_song_dict
import matplotlib.pyplot as plt
from sklearn import preprocessing
import cPickle as pickle
import yaml
from os import listdir
from os.path import basename, splitext

def load_all_yaml(dir_, feature_names):
    print 'loading YAML files...'

    nb_loaded_files = 0
    d_all_essentia = dict()
    for yaml_filename in listdir(dir_):
        nb_loaded_files += 1

        song_id = int(splitext(basename(yaml_filename))[0])
        with open(dir_ + yaml_filename, 'r') as stream:
            d_data = yaml.load(stream)
        first = True
        X = np.array([])
        for featname in feature_names:
            # print featname
            tmp = np.array(d_data['lowlevel'][featname])
            if len(tmp.shape) == 1:
                tmp = tmp[:, np.newaxis]
            # tmp = tmp[0:NUM_FRAMES, :]
            # remove first two central moments (std = 0)
            if featname is 'central_moments_bark' or featname is 'central_moments_erb' or featname is 'central_moments_mel':
                tmp = tmp[:, 2:]
            # print 'shape(tmp): ', tmp.shape
            if first:
                X = tmp
                first = False
                # print X.shape
            else:
                X = np.hstack((X, tmp))
                # print X.shape

        d_all_essentia['%d'%(song_id)] = X

        if nb_loaded_files % 10 == 0:
            print ' loaded: %d files'%(nb_loaded_files)

    return d_all_essentia

if __name__ == '__main__':
    # NUM_FRAMES = 60
    NUM_OUTPUT = 2
    DATADIR = '/baie/corpus/emoMusic/test/'
    # DATADIR = './test/'
    ESSENTIA_DIR = DATADIR + 'essentia_features/'

    metadatafile = DATADIR + 'metadata_testset_2015.csv'


    doUseEssentiaFeatures = True
    feature_names= [ 'flatnessdb_bark', 'flatnessdb_erb', 'spectral_valley' ]

    # keys: song ids, values; dict with song_data_dict[song_id]['X'] = np 2d array
    baseline_song_data_dict = load_TEST_data_to_song_dict(metadatafile, DATADIR)

    if doUseEssentiaFeatures:
        # load yaml files
        d_all_essentia = load_all_yaml(ESSENTIA_DIR, feature_names)
        nb_features = 268
    else:
        nb_features = 260

    # varying duration -> not possible to create a 3d tensor
    #

    # for k, v in song_data_dict.iteritems():
    #     print k, len(v['X'])
    # v['X'] are 2d np arrays

    # print '... create a single 3d tensor ...'
    # # pick an item to get dimensions
    # tmp = song_data_dict.itervalues().next()
    # frame_dim, feature_dim = tmp['X'].shape
    # nb_of_items = len(song_data_dict)
    # X_ = np.zeros((nb_of_items, frame_dim, feature_dim), dtype = float)

    song_ids = baseline_song_data_dict.keys()

    if doUseEssentiaFeatures:
        # add essentia features
        song_data_dict = dict()

        for id in song_ids:
            baseline_X = baseline_song_data_dict[id]['X']
            essentia_X = d_all_essentia[id]
            taille = min(baseline_X.shape[0], essentia_X.shape[0])
            print taille, baseline_X.shape, essentia_X.shape
            song_data_dict[id] = dict()
            song_data_dict[id]['X'] = np.concatenate((baseline_X[0:taille, :], essentia_X[0:taille, :]), axis=1)
    else:
        song_data_dict = baseline_song_data_dict

    print '... saving to PKL file ... '
    # data = dict()
    # data['test'] = dict()
    # data['test']['X'] = X_
    # data['test']['song_id'] = song_ids

    nom = DATADIR + 'pkl/test_set_baseline_%dfeatures_58songs_NOT_normed.pkl'%(nb_features)

    pickle.dump( song_data_dict, open( nom, "wb" ) )
    print ' ... --> saved to: ** %s **'%(nom)

    print '\n... standardizing data ... '

    # X_2d = np.reshape(X_, (X_.shape[0]*X_.shape[1], X_.shape[2]), order='C')

    scaler = preprocessing.StandardScaler()

    # load means and variances from the training dataset
    train_file = '/baie/corpus/emoMusic/train/pkl/train_set_baseline_%dfeatures_431songs_normed.pkl'%(nb_features)
    train_data = pickle.load( open( train_file, "rb" ) )

    means = train_data['train']['mean']
    stds = train_data['train']['std']

    scaler.mean_ = means
    scaler.std_ = stds

    normed_song_data_dict = dict()

    for id, v in song_data_dict.iteritems():
        X_ = v['X']
        X_normed = scaler.transform(X_)

        normed_song_data_dict[id] = dict()
        normed_song_data_dict[id]['X'] = X_normed
        # print id
        # print 'before: '
        # print np.mean(X_, axis=0)
        # print np.std(X_, axis=0)
        # print 'after: '
        # print np.mean(X_normed, axis=0)
        # print np.std(X_normed, axis=0)



    # data = dict()
    # data['test'] = dict()
    # data['test']['X'] = X_normed
    # data['test']['song_id'] = song_ids
    nom = DATADIR + '/pkl/test_set_baseline_%dfeatures_58songs_normed.pkl'%(nb_features)
    pickle.dump( normed_song_data_dict, open( nom, "wb" ) )
    print ' ... --> saved to: ** %s **'%(nom)

