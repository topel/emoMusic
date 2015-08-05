__author__ = 'tpellegrini'

import numpy as np
from utils import load_data_to_song_dict
# import matplotlib.pyplot as plt
from sklearn import preprocessing
import cPickle as pickle
import yaml
from os import listdir
from os.path import basename, splitext

def load_all_yaml(dir_, NUM_FRAMES):
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
            tmp = tmp[0:NUM_FRAMES, :]
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

        if nb_loaded_files % 100 == 0:
            print ' loaded: %d files'%(nb_loaded_files)

    return d_all_essentia

if __name__ == '__main__':
    NUM_FRAMES = 60
    NUM_OUTPUT = 2
    DATADIR = '/baie/corpus/emoMusic/train/'
    # DATADIR = './train/'
    ESSENTIA_DIR = DATADIR + 'essentia_features/'
    doUseEssentiaFeatures = True

    # feature_names= ['loudness',
    #                 'spectrum_rms','spectrum_flux','spectrum_centroid','spectrum_rolloff', 'spectrum_decrease',
    #                 'hfc','zcr',
    #                 'mfcc','mfcc_bands',
    #                 'barkbands', 'crest_bark', 'flatnessdb_bark', 'central_moments_bark',
    #                 'erbbands', 'crest_erb', 'flatnessdb_erb','central_moments_erb',
    #                 'melbands','crest_mel','flatnessdb_mel','central_moments_mel',
    #                 'gfcc','spectral_contrast','spectral_valley','dissonance','pitchsalience','spectral_complexity',
    #                 'danceability']

    feature_names= [ 'flatnessdb_bark', 'flatnessdb_erb', 'spectral_valley' ]

    # not used: 'dynamic_complexity', first two elements of 'central_moments_bark', 'central_moments_erb', 'central_moments_mel'

    if doUseEssentiaFeatures:
        # load yaml files
        d_all_essentia = load_all_yaml(ESSENTIA_DIR, NUM_FRAMES)

    metadatafile = DATADIR + 'annotations/metadata.csv'
    list_genres_of_interest_file = DATADIR + 'annotations/categories.lst'
    severalGenresPerSong = True

    song_data_dict = load_data_to_song_dict(metadatafile, list_genres_of_interest_file, DATADIR, severalGenresPerSong)

    ### plot arousal = f ( valence )
    # songid = 732
    # plot_valence_arousal(song_data_dict, songid)
    print '... create a single 3d tensor ...'
    # pick an item to get dimensions
    tmp = song_data_dict.itervalues().next()
    frame_dim, feature_dim = tmp['X'].shape
    nb_of_items = len(song_data_dict)
    X_ = np.zeros((nb_of_items, frame_dim, feature_dim), dtype = float)
    y_ = np.zeros((nb_of_items, frame_dim, NUM_OUTPUT), dtype = float)

    song_ids = list()
    ind_sequence = 0
    for k, v in song_data_dict.iteritems():
        if (k == 'std' or k == 'mean'):
            continue
        song_ids.append(k)
        val = np.array(v['valence'], dtype=float)
        ar = np.array(v['arousal'], dtype=float)

        X_[ind_sequence] = v['X']
        y_[ind_sequence] = np.hstack((val[:,np.newaxis], ar[:,np.newaxis]))
        ind_sequence += 1

    if doUseEssentiaFeatures:
        # add essentia features
        essentia_X_ = list()
        for id in song_ids:
            essentia_X_.append(d_all_essentia[id])
        essentia_X_ = np.array(essentia_X_)
        X_ = np.concatenate((X_, essentia_X_), axis=2)

    print X_.shape

    song_ids = np.array(song_ids, dtype=int)

    print '... saving to PKL file ... '
    data = dict()
    data['train'] = dict()
    data['train']['X'] = X_
    data['train']['y'] = y_
    data['train']['song_id'] = song_ids
    if doUseEssentiaFeatures:
        nom = DATADIR + '/pkl/train_set_baseline_268features_431songs_NOT_normed.pkl'
    else:
        nom = DATADIR + '/pkl/train_set_baseline_260features_431songs_NOT_normed.pkl'

    pickle.dump( data, open( nom, "wb" ) )
    print ' ... --> saved to: ** %s **'%(nom)

    print '\n... standardizing data ... '

    X_2d = np.reshape(X_, (X_.shape[0]*X_.shape[1], X_.shape[2]), order='C')

    scaler = preprocessing.StandardScaler().fit(X_2d)

    data = dict()
    data['train'] = dict()
    data['train']['mean'] = scaler.mean_
    data['train']['std'] = scaler.std_

    X_2d_normed = scaler.transform(X_2d)

    X_normed = np.reshape(X_2d_normed, (X_.shape[0], X_.shape[1], X_.shape[2]), order='C')

    data['train']['X'] = X_normed
    data['train']['y'] = y_
    data['train']['song_id'] = song_ids
    if doUseEssentiaFeatures:
        nom = DATADIR + '/pkl/train_set_baseline_268features_431songs_normed.pkl'
    else:
        nom = DATADIR + '/pkl/train_set_baseline_260features_431songs_normed.pkl'

    pickle.dump( data, open( nom, "wb" ) )
    print ' ... --> saved to: ** %s **'%(nom)
