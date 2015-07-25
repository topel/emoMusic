__author__ = 'thomas'

import csv
import numpy as np
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
import subprocess
from utils import load_y_to_dict
import time
from os import path
import cPickle as pickle
from scipy.stats import pearsonr
from sklearn import preprocessing

feature_dir='/baie/corpus/emoMusic/train/vamp_features'
tempo_dir = feature_dir + '/tempo'
DATADIR = '/baie/corpus/emoMusic/train/'
NUM_FRAMES = 60

data = dict()
arousal, valence, song_ids, nb_of_songs = load_y_to_dict(DATADIR)

tmp_array = np.zeros((nb_of_songs*NUM_FRAMES, 3))
ind = 0

# song_id=250
for song_id in song_ids:
# for song_id in [3, 4]:
    # file='/home/thomas/software/sonic-annotator-1.1/250_vamp_mtg-melodia_melodia_melody.csv'
    file = tempo_dir + '/%d_vamp_vamp-example-plugins_fixedtempo_tempo.csv'%(song_id)

    tempo = np.array([])
    with open(file, 'r') as csvfile:
        temporeader = csv.reader(csvfile)
        for row in temporeader:
            # tempo.append(np.float(row[2]) * np.ones(NUM_FRAMES))
            tempo = np.float(row[2]) * np.ones(NUM_FRAMES)

    # print np.array(arousal['%d'%(song_id)])
    tmp_array[ind:ind+NUM_FRAMES,0] = np.array(arousal['%d'%(song_id)])
    tmp_array[ind:ind+NUM_FRAMES:,1] = np.array(valence['%d'%(song_id)])
    tmp_array[ind:ind+NUM_FRAMES,2] = np.array(tempo)
    # print np.array(tempo)

    data['%d'%song_id] = dict()
    data['%d'%song_id]['fixed_tempo'] = tempo
    # print tempo.shape
    ind += NUM_FRAMES

print tmp_array.shape
ar = tmp_array[:,0]
val = tmp_array[:,1]
tempos = tmp_array[:,2]

print tempos.shape

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
scaled_tempos = min_max_scaler.fit_transform(np.array([tempos]).T)
# scaled_tempos = min_max_scaler.transform([tempos])
print scaled_tempos.shape

print pearsonr(ar, tmp_array[:,2])
print pearsonr(val, tmp_array[:,2])
print pearsonr(ar, val)

plt.plot(tmp_array[:,0])
plt.plot(tmp_array[:,1])
plt.plot(scaled_tempos[:,0])
plt.show()

# nom = DATADIR + '/pkl/fixedtempo_features.pkl'
# pickle.dump( data, open( nom, "wb" ) )
# print ' ... output file: %s'%(nom)

doPlot = False
if doPlot:
    # plot val, ar, slopes and smoothed slopes for a song, here the 250th song
    # val * 1000
    fig, ax = plt.subplots(2,1)

    # ax[0].plot(1000.*val, label='valence (x10^3)')
    # ax[0].plot(1000.*ar, label='arousal (x10^3)')
    ax[0].plot(val, label='valence')
    ax[0].plot(ar, label='arousal')
    # ax[0].plot(slopes[delay_slopes:], label = 'mel slopes r_val=%.3f, r_ar=%.3f'%(corr_val, corr_ar))
    # ax[0].plot(slopes_f, label = 'mel slopes r_val=%.3f, r_ar=%.3f'%(corr_val_f, corr_ar_f))
    # ax[0].plot(slopes_f2, label = 'mel slopes delayed')
    legend = ax[0].legend(loc='upper left', shadow=True, fontsize=11)
    # legend= ax[0].legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    # ax[1].plot((mel_start15s[:,0]-15.)*2., mel_start15s[:,1], label = 'melody')
    ax[1].plot(tempos, label = 'tempo')
    # ax[1].plot(1000.*(val+1), label='valence+1 (x10^3)')
    # ax[1].plot(1000.*(ar+1), label='arousal+1 (x10^3)')
    # ax[1].plot(slopes[delay_slopes:], label = 'mel slopes r_val=%.3f, r_ar=%.3f'%(corr_val, corr_ar))
    # ax[1].plot(slopes_f, label = 'mel slopes smoothed r_val=%.3f, r_ar=%.3f'%(corr_val_f, corr_ar_f))
    # legend = ax[1].legend(loc='upper center', shadow=True, fontsize=11)
    legend = ax[1].legend(loc='upper center', shadow=True, fontsize=11)
    # plt.xlim([0,60])
    # fig.suptitle('song %d'%(song_id), fontsize=30)
    plt.show()
    # plt.savefig('val_ar_melody_song%d_smooth.png'%song_id)
