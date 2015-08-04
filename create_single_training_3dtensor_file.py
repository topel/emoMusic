__author__ = 'thomas'

__author__ = 'tpellegrini'

import numpy as np
from utils import load_data_to_song_dict, create_folds, standardize_folds, standardize, add_intercept
import matplotlib.pyplot as plt
from sklearn import preprocessing
import cPickle as pickle

def plot_valence_arousal(song_data_dict, songid=3):

    song = song_data_dict['%d'%songid]
    val = song['valence']
    ar = song['arousal']

    plt.plot(val, ar, 'o')
    plt.title('song: %d, genre: %s'%(songid, song['genre']))
    axes = plt.gca()
    axes.set_xlim([-1.,1.])
    axes.set_ylim([-1.,1.])
    plt.show()



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

def write_folds_to_pickle_files(normed_folds, num_folds, DATADIR, doNormalize):

    import cPickle as pickle
    for fold in xrange(num_folds):
        print ' ... creating pickle file for fold: %d ...'%(fold)

        data = dict()
        data['train'] = normed_folds[fold][0]
        data['test'] = normed_folds[fold][1]

        # save to pickle file
        if doNormalize:
            nom = DATADIR + '/pkl/fold%d_normed.pkl'%(fold)
            pickle.dump( data, open( nom, "wb" ) )
            print ' ... output file: %s'%(nom)
        else:
            nom = DATADIR + '/pkl/fold%d_NOT_normed.pkl'%(fold)
            pickle.dump( data, open( nom, "wb" ) )
            print ' ... output file: %s'%(nom)

if __name__ == '__main__':
    NUM_FRAMES = 60
    NUM_OUTPUT = 2
    DATADIR = '/baie/corpus/emoMusic/train/'
    # DATADIR = './train/'
    doNormalize = False

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

    song_ids = np.array(song_ids, dtype=int)

    print '... saving to PKL file ... '
    data = dict()
    data['train'] = dict()
    data['train']['X'] = X_
    data['train']['y'] = y_
    data['train']['song_id'] = song_ids
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
    nom = DATADIR + '/pkl/train_set_baseline_260features_431songs_normed.pkl'
    pickle.dump( data, open( nom, "wb" ) )
    print ' ... --> saved to: ** %s **'%(nom)
