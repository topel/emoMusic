__author__ = 'tpellegrini'

import numpy as np
from utils import load_data_to_song_dict, create_folds, standardize_folds, standardize, add_intercept
import matplotlib.pyplot as plt

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
        ' ... creating MAT file for fold: %d ...'%(fold)
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

def write_folds_to_pickle_files(normed_folds, num_folds, DATADIR):

    import cPickle as pickle
    for fold in xrange(num_folds):
        ' ... creating pickle file for fold: %d ...'%(fold)

        data = dict()
        data['train'] = normed_folds[fold][0]
        data['test'] = normed_folds[fold][1]

        # save to pickle file
        pickle.dump( data, open( DATADIR + '/pkl/fold%d_normed.pkl'%(fold), "wb" ) )


if __name__ == '__main__':
    NUM_FRAMES = 60
    DATADIR = '/baie/corpus/emoMusic/train/'
    # DATADIR = './train/'

    metadatafile = DATADIR + 'annotations/metadata.csv'
    list_genres_of_interest_file = DATADIR + 'annotations/categories.lst'
    severalGenresPerSong = True

    song_data_dict = load_data_to_song_dict(metadatafile, list_genres_of_interest_file, DATADIR, severalGenresPerSong)

    ### plot arousal = f ( valence )
    # songid = 732
    # plot_valence_arousal(song_data_dict, songid)

    num_folds = 10
    folds = create_folds(song_data_dict, num_folds)
    # print len(folds[0][0]), len(folds[0][1])

    print '... normalizing folds ...'
    normed_folds = standardize_folds(folds)

    # print '... writing folds to MAT files ...'
    # write_folds_to_mat_files(normed_folds, num_folds)

    print '... writing folds to pickle files ...'
    write_folds_to_pickle_files(normed_folds, num_folds, DATADIR)

    import cPickle as pickle
    fold_id = 0
    fold0 = pickle.load( open( DATADIR + '/pkl/fold%d_normed.pkl'%(fold_id), "rb" ) )
