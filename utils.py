__author__ = 'thomas'

import csv
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats.stats import pearsonr

def load_y(datadir):
    FILENAME = datadir + 'annotations/dynamic_arousals.csv'
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

def load_X(datadir, song_id):
    # Number of frame per song
    NUM_FRAMES = 60

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

def mix(X, y, purcent, num_frames, song_id, nb_of_songs):
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
    # # MSE = np.mean(np.sqrt((y_hat - y_test) ** 2),axis=0)
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

