__author__ = 'thomas'

import csv
import numpy as np
import random
from sklearn.metrics import mean_squared_error
from scipy.stats.stats import pearsonr
from sklearn import preprocessing
import re
from copy import deepcopy
import subprocess

def load_y(datadir):
    FILENAME = datadir + 'annotations/dynamic_arousals.csv'
    print '... loading y: ', FILENAME
    song_id = []
    y_temp = {}
    Index = 0
    with open(FILENAME, "rb") as infile:
        reader = csv.reader(infile)
        next(reader, None)  # skip the headers
        for row in reader:
            song_id.append(int(row[0]))
            y_temp[Index,0] = row[1::]
            Index += 1

    FILENAME = datadir + 'annotations/dynamic_valences.csv'
    print '... loading y: ', FILENAME
    Index = 0
    with open(FILENAME, "rb") as infile:
        reader = csv.reader(infile)
        next(reader, None)  # skip the headers
        for row in reader:
            y_temp[Index,1] = row[1::]
            Index += 1

    nb_of_songs = len(song_id)
    num_col = len(y_temp[0,0])

    y_temp_array = np.array([ [[float(y_temp[k,l][m]) for m in range(num_col) ] for k in range(nb_of_songs)] for l in [0,1] ])

    # print y_temp_array[0:3,0,:]

    # To transform the y in the good shape
    y_buff_arr = y_temp_array[0,0,:]
    y_buff_val = y_temp_array[1,0,:]
    for k in range(1,y_temp_array.shape[1]):
        y_buff_arr = np.hstack((y_buff_arr,y_temp_array[0,k,:]))
        y_buff_val = np.hstack((y_buff_val,y_temp_array[1,k,:]))

    y = np.vstack((y_buff_arr,y_buff_val)).transpose(1,0)

    return y,song_id,  nb_of_songs



def load_y_to_dict(datadir):
    FILENAME = datadir + 'annotations/dynamic_arousals.csv'
    print '... loading y: ', FILENAME
    song_id = []
    arousal = dict()
    valence = dict()
    with open(FILENAME, "rb") as infile:
        reader = csv.reader(infile)
        next(reader, None)  # skip the headers
        for row in reader:
            id = int(row[0])
            song_id.append(id)
            arousal['%d'%id] = [float(val) for val in row[1::]]

    FILENAME = datadir + 'annotations/dynamic_valences.csv'
    print '... loading y: ', FILENAME
    Index = 0
    with open(FILENAME, "rb") as infile:
        reader = csv.reader(infile)
        next(reader, None)  # skip the headers
        for row in reader:
            id = int(row[0])
            valence['%d'%id] = [float(val) for val in row[1::]]

    nb_of_songs = len(song_id)
    return arousal, valence, song_id,  nb_of_songs

def load_X_to_dict(datadir, song_id):
    # Number of frame per song
    NUM_FRAMES = 60
    print '... loading X to dict, keys=song_id, values=feature_vector'
    X = dict()
    for Id in song_id:
        FILENAME = datadir + "openSMILE_features/%d.csv" %Id

        with open(FILENAME, 'rb') as infile:
            reader = csv.reader(infile, delimiter =";")
            next(reader, None)  # skip the headers
            X_temp = [ row for row in reader ]

        num_features = len(X_temp[0])
        num_col = len(X_temp)

        X_temp_array = np.array([ [float(X_temp[k][l]) for l in range(1,num_features) ] for k in range(num_col-NUM_FRAMES,num_col)] )

        X['%d'%Id] = X_temp_array

    return X


def load_X(datadir, song_id):
    # Number of frame per song
    NUM_FRAMES = 60
    print '... loading X '
    X = None
    for Id in song_id:
        FILENAME = datadir + "openSMILE_features/%d.csv" %Id

        with open(FILENAME, 'rb') as infile:
            reader = csv.reader(infile, delimiter =";")
            next(reader, None)  # skip the headers
            X_temp = [ row for row in reader ]

        num_features = len(X_temp[0])
        num_col = len(X_temp)

        X_temp_array = np.array([ [float(X_temp[k][l]) for l in range(1,num_features) ] for k in range(num_col-NUM_FRAMES,num_col)] )
        if X is None:
            X = X_temp_array
        else:
            X = np.vstack((X,X_temp_array))
    return X

def load_metadata(metadatafile, genrelistfile, severalGenresPerSong):
    '''load genre info of songs
    - genre names are taken from the list in genrelistfile

    - inputs:
        - metadatafile: csv file
        - genrelistfile: text file
        - severalGenresPerSong: boolean to search for several genres per song
        example: raw genre: 'rock-classical' -> two genres: 'rock' and 'classical'
    - outputs:
        - genre: a dict with keys=song_id, values=a single genre string or a list of strings with one or more genres of the song'
        - genrenum: the same dict but with integer as genre indexes instead of strings
    '''

    genre_of_interest = list()
    genre_2_num = dict()
    ind = 0
    with open(genrelistfile, "rb") as infile:
        reader = csv.reader(infile)
        for row in reader:
            genre_of_interest.append(row[0])
            genre_2_num[row[0]] = ind
            ind += 1

    genre = dict()
    genre_num = dict()
    with open(metadatafile, "rb") as infile:
        reader = csv.reader(infile)
        next(reader, None)  # skip the headers
        if not severalGenresPerSong:
            for row in reader:
                id = int(row[0])
                genre['%d'%id] = row[-1]

        else:
            for row in reader:
                raw_genre = row[-1]
                Id = int(row[0])
                genre['%d'%Id] = list()
                genre_num['%d'%Id] = list()
                for g in genre_of_interest:
                    if re.search(g, raw_genre):
                        genre['%d'%Id].append(g)
                        genre_num['%d'%Id].append(genre_2_num[g])
                if len(genre['%d'%Id]) < 1:
                    print 'ERROR: no genre label for song id: %d'%(Id)

    return genre, genre_num, genre_of_interest


def create_song_dict(X, arousal, valence, song_id, genre, genre_num):
    data = dict()
    for id in song_id:
        # print id, genre['%d'%id]
        data['%d'%id] = dict()
        data['%d'%id]['genre'] = genre['%d'%id]
        data['%d'%id]['genrenum'] = genre_num['%d'%id]
        data['%d'%id]['X'] = X['%d'%id]
        data['%d'%id]['arousal'] = arousal['%d'%id]
        data['%d'%id]['valence'] = valence['%d'%id]
        # print id, data['%d'%id]['genre'], data['%d'%id]['X'][0:2,0:5], data['%d'%id]['arousal'][0:5], data['%d'%id]['valence'][0:5]
        # print id, data['%d'%id]['genre']
    return data


def load_data_to_song_dict(metadatafile, genrelistfile, datadir, severalGenresPerSong):
    print 'loading metadata...'
    genre, genre_num, genre_of_interest = load_metadata(metadatafile, genrelistfile, severalGenresPerSong)
    print 'loading aoursal, valence, data...'
    arousal, valence, song_id, nb_of_songs = load_y_to_dict(datadir)
    print 'loading X...'
    X = load_X_to_dict(datadir, song_id)
    print 'creating dict with keys=song_ids, values=genre, arousal, valence, X'
    data = create_song_dict(X, arousal, valence, song_id, genre, genre_num)

    return data

def load_TEST_data_to_song_dict(metadatafile, datadir):

    print 'loading song ids...'

    test_metadata_dict = dict()
    # keys=song_ids, values=filenames
    with open(metadatafile, "rb") as infile:
        reader = csv.reader(infile)
        next(reader, None)  # skip the headers
        for row in reader:
            id = int(row[0])
            test_metadata_dict['%d'%id] = row[1]


# # create symbolic links to have song ids as file names:
#     for id, fn in test_metadata_dict.iteritems():
#         print 'linking %s to %s'%(fn, id)
#         nom = datadir + 'openSMILE_features/' + fn + '.csv'
#         new_nom = datadir + 'openSMILE_features/' + id + '.csv'
#         cline = ['ln', '-s', nom, new_nom]
#         subprocess.call(cline)

    song_id = list()
    [song_id.append(int(k)) for k in test_metadata_dict.keys()]

    print 'loading X...'
    X = load_X_to_dict(datadir, song_id)
    print 'creating dict with keys=song_ids, values=X'
    data = dict()
    for id in song_id:
        # print id, genre['%d'%id]
        data['%d'%id] = dict()
        data['%d'%id]['X'] = X['%d'%id]

    return data



def create_folds(data, num_folds):
    '''Create num_folds training and testing sets'''
    subsets = []
    keys = list(data.keys())
    samples_per_fold = len(data) / num_folds
    for i in xrange(num_folds):
        # Get the number of samples for this fold
        # num_samples = samples_per_fold + 1 if i < len(data) % num_folds else samples_per_fold

        # Sample without replacement from the available keys

        np.random.seed(i) # to get reproducible results
        selected = np.random.choice(keys, size=samples_per_fold, replace=False)

        # Remove the keys from available set
        for s in selected:
            keys.remove(s)

        # Add the selected subsets to the fold
        subsets.append({s: data[s] for s in selected})

    # Create training and testing sets from the folds
    folds = []

    # Note that this is terrible big-O, but folds are generally small so who cares
    for i,testing in enumerate(subsets):
        training = {}
        for j,fold in enumerate(subsets):
            if i == j:
                continue
            for key, value in fold.iteritems():
                training[key] = value
        folds.append((training, testing))

    return folds

def standardize_folds(folds):
    '''see http://scikit-learn.org/stable/modules/preprocessing.html'''
    new_folds = deepcopy(folds)

    for fold in xrange(len(folds)):
        print '... standardizing fold: %d ... ' %(fold)
        train = folds[fold][0]
        feat = None
        for _, val in train.iteritems():
            if feat is None:
                feat = val['X']
            else:
                feat = np.vstack((feat, val['X']))

        scaler = preprocessing.StandardScaler().fit(feat)
        new_folds[fold][0]['mean'] = scaler.mean_
        new_folds[fold][0]['std'] = scaler.std_

        # standardize training data
        for cle, val in train.iteritems():
            new_folds[fold][0][cle]['X'] = scaler.transform(val['X'])

        # standardize test data
        test = folds[fold][1]
        for cle, val in test.iteritems():
            new_folds[fold][1][cle]['X'] = scaler.transform(val['X'])

        # break


    return new_folds


def load_X_from_fold(fold, subset):
    song_ids = list()
    X_ = None
    y_ = None
    for k, v in fold[subset].iteritems():
        if (k == 'std' or k == 'mean'):
            continue
        song_ids.append(k)
        val = np.array(v['valence'], dtype=float)
        ar = np.array(v['arousal'], dtype=float)
        tmp = np.hstack((val[:,np.newaxis], ar[:,np.newaxis]))

        if X_ is None:
            X_ = v['X']
            y_ = tmp
        else:
            X_ = np.vstack((X_, v['X']))
            y_ = np.vstack((y_, tmp))
    return X_,  y_, np.array(song_ids, dtype=int)

def load_X_from_fold_to_3dtensor(fold, subset, out_dim):
    song_ids = list()
    # nb of samples in subset: we substract 2 for the 'mean' and 'std' items
    if subset == 'train':
        nb_of_items = len(fold[subset]) - 2
    else:
        nb_of_items = len(fold[subset])

    # print nb_of_items

    # pick an item to get dimensions
    tmp = fold[subset].itervalues().next()
    frame_dim, feature_dim = tmp['X'].shape
    X_ = np.zeros((nb_of_items, frame_dim, feature_dim), dtype = float)
    y_ = np.zeros((nb_of_items, frame_dim, out_dim), dtype = float)


    ind_sequence = 0
    for k, v in fold[subset].iteritems():
        if (k == 'std' or k == 'mean'):
            continue
        song_ids.append(k)
        val = np.array(v['valence'], dtype=float)
        ar = np.array(v['arousal'], dtype=float)

        X_[ind_sequence] = v['X']
        y_[ind_sequence] = np.hstack((val[:,np.newaxis], ar[:,np.newaxis]))
        ind_sequence += 1
    return X_,  y_, np.array(song_ids, dtype=int)

def load_data_from_fold_to_3dtensor_and_genre_info(fold, subset, out_dim, expected_nb_of_genres):
    song_ids = list()
    # nb of samples in subset: we substract 2 for the 'mean' and 'std' items
    # nb_of_items = len(fold[subset]) - 2
    # pick an item to get dimensions
    tmp = fold[subset].itervalues().next()
    frame_dim, feature_dim = tmp['X'].shape

    list_num_genre = list()
    for k, v in fold[subset].iteritems():
        if (k == 'std' or k == 'mean'):
            continue
        list_num_genre.append(v['genrenum'])

    flatten_list_num_genre = [item for sublist in list_num_genre for item in sublist]
    nb_of_items =len(flatten_list_num_genre)

    set_num_genre = set(flatten_list_num_genre)
    nb_of_genre = len(set_num_genre)
    if nb_of_genre < expected_nb_of_genres:
        nb_of_genre = expected_nb_of_genres

    X_ = np.zeros((nb_of_items, frame_dim, feature_dim), dtype = float)
    y_ = np.zeros((nb_of_items, frame_dim, out_dim), dtype = float)
    genre_indexes = np.zeros((nb_of_genre, 2), dtype = int)

# pour python: indexes begin 0
    debut = 0

# pour matlab: indexes begin 1
    debut = 1
    ind_sequence = 0
    for genre in set_num_genre:
        nb_seq_per_genre = 0
        # print 'genre: %d'%(genre)
        first_time = True
        for k, v in fold[subset].iteritems():
            if (k == 'std' or k == 'mean'):
                continue

            seq_genres = v['genrenum']

            for seq_g in seq_genres:
                if genre == seq_g:
                    # print 'genre: %d, genrenum: %d, genre_str: %s, ind: %d, song_id: %s'%(genre, seq_g, v['genre'], ind_sequence, k)
                    if first_time:
                        genre_indexes[genre,0] = ind_sequence
                        first_time = False
                    song_ids.append(k)
                    val = np.array(v['valence'], dtype=float)
                    ar = np.array(v['arousal'], dtype=float)

                    X_[ind_sequence,:,:] = v['X']
                    y_[ind_sequence,:,:] = np.hstack((val[:,np.newaxis], ar[:,np.newaxis]))
                    ind_sequence += 1
                    nb_seq_per_genre += 1
        print 'genre: %d nb_seqs: %d'%(genre, nb_seq_per_genre)
        genre_indexes[genre,0] = debut
        genre_indexes[genre,1] = debut + nb_seq_per_genre - 1
        debut += nb_seq_per_genre

    return X_,  y_, np.array(song_ids, dtype=int), genre_indexes

def standardize(X, scaler=None):
    '''see http://scikit-learn.org/stable/modules/preprocessing.html'''
    if scaler == None:
        scaler = preprocessing.StandardScaler().fit(X)
        print 'standardizing w/o scaler'
    else:
        print 'standardizing with scaler'
    X = scaler.transform(X)
    return X, scaler

def add_intercept(X):
    # add column of ones to data to account for the bias:
    ones = np.ones((X.shape[0],1))
    # print ones.shape
    X_ = np.hstack((X, ones))
    return X_

def mix(X, y, purcent, num_frames, song_id, nb_of_songs):
    print '... subsetting and shuffle'
    # Vector of permutation of length the number of song (indicating the coordonnate of the begginning of each song)
    seed = 1
    np.random.seed(seed)
    mix = np.random.permutation(nb_of_songs)*num_frames

    # mix vector per song for all the samples
    mix_finish = np.array([range(mix[0],mix[0]+num_frames)])
    for k in range(1,nb_of_songs):
        mix_finish = np.hstack(( mix_finish, np.array([range(mix[k],mix[k]+num_frames)]) ))

    X_rand = X[mix_finish][0,:,:]
    y_rand = y[mix_finish][0,:,:]

    # Purcent of the set you want on the test set
    tst_song = int(purcent*nb_of_songs/100.)
    tst = tst_song*num_frames

    # Split the data into training/testing sets
    X_train = X_rand[:-tst]
    X_test = X_rand[-tst:]

    # Split the targets into training/testing sets
    y_train = y_rand[:-tst]
    y_test = y_rand[-tst:]

    # Which song is it in the test_set ?
    song_id_tst = int(song_id[0])
    for k in range(1,nb_of_songs):
        song_id_tst = np.hstack((song_id_tst,int(song_id[k])))

    #song_id_tst = np.array([[song_id[0]]],dtype = int)
    #for k in range(1,nb_of_songs):
    #    song_id_tst = np.vstack((song_id_tst,np.array([[song_id[k]]],dtype = int)))

    song_id_tst = song_id_tst[mix[-tst_song:]/num_frames]

    return X_train, y_train, X_test, y_test, song_id_tst

def evaluate(y_test, y_hat, tst_song):
    # # The mean square error
    # MSE = np.mean((y_hat - y_test) ** 2, axis=0)
    # RMSE = np.sqrt(MSE)

    diff = np.sqrt((y_hat - y_test)**2)
    error_per_song = np.array([np.mean(diff[60*k:60*(k+1)],axis=0) for k in range(tst_song)])
    mean_per_song = np.array([np.mean(y_test[60*k:60*(k+1)],axis=0) for k in range(tst_song)])

    RMSE = list()
    RMSE.append(mean_squared_error(y_test[:,0], y_hat[:,0])**0.5)
    RMSE.append(mean_squared_error(y_test[:,1], y_hat[:,1])**0.5)

    pcorr = list()
    pcorr.append(pearsonr(y_test[:,0], y_hat[:,0]))
    pcorr.append(pearsonr(y_test[:,1], y_hat[:,1]))

    return RMSE, pcorr, error_per_song, mean_per_song

def evaluate1d(y_test, y_hat, tst_song):
    # # The mean square error
    # MSE = np.mean((y_hat - y_test) ** 2, axis=0)
    # # MSE = np.mean(np.sqrt((y_hat - y_test) ** 2),axis=0)
    # RMSE = np.sqrt(MSE)

    diff = np.sqrt((y_hat - y_test)**2)
    error_per_song = np.array([np.mean(diff[60*k:60*(k+1)],axis=0) for k in range(tst_song)])
    mean_per_song = np.array([np.mean(y_test[60*k:60*(k+1)],axis=0) for k in range(tst_song)])

    RMSE = list()
    RMSE.append(mean_squared_error(y_test[:], y_hat[:])**0.5)

    pcorr = list()
    pcorr.append(pearsonr(y_test[:], y_hat[:]))

    return RMSE, pcorr, error_per_song, mean_per_song

def subset_features(feature_dict, song_id_train, song_id_test):

    # allocate feat_train and feat_test
    nb_seq_train = len(song_id_train)
    nb_seq_test = len(song_id_test)
    nb_features = 0
    frame_dim = 60

    # to get dimensions, pick up a single song
    one_song = feature_dict.itervalues().next()
    for k, v in one_song.iteritems():
        sh_ = v.shape
        if (len(sh_)  == 2):
            frame_dim, feature_dim = v.shape
        else:
            feature_dim = 1
        nb_features += feature_dim

    print 'frame_dim=%d, nb_features=%d'%(frame_dim, nb_features)
    # allocate result arrays
    feat_train = np.zeros((nb_seq_train, frame_dim, nb_features))
    feat_test = np.zeros((nb_seq_test, frame_dim, nb_features))

    # fulfill them
    train_ind = 0
    for id in song_id_train:
        feat_dict_per_song = feature_dict['%d'%id]
        tmp = None
        for _, feat in feat_dict_per_song.iteritems():
            if tmp == None:
                if len(feat.shape) == 1:
                    feat = feat[:, np.newaxis]
                tmp = feat
            else:
                if len(feat.shape) == 1:
                    feat = feat[:, np.newaxis]
                tmp = np.hstack((tmp, feat))
        feat_train[train_ind] = tmp
        train_ind += 1

    test_ind = 0
    for id in song_id_test:
        feat_dict_per_song = feature_dict['%d'%id]
        tmp = None
        for _, feat in feat_dict_per_song.iteritems():
            if tmp == None:
                if len(feat.shape) == 1:
                    feat = feat[:, np.newaxis]
                tmp = feat
            else:
                if len(feat.shape) == 1:
                    feat = feat[:, np.newaxis]
                tmp = np.hstack((tmp, feat))
        feat_test[test_ind] = tmp
        test_ind += 1

    return feat_train, feat_test