__author__ = 'thomas'

import csv
import numpy as np
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
from utils import load_y_to_dict
from scipy.stats import pearsonr

def compute_lin_reg(frame):
    '''
    :param frame: np array with col1 = time, col2 = feature values
    :return: slope and intercept terms
    '''
    sxy=np.dot(frame[:,0], frame[:,1].T)
    sx = np.sum(frame[:,0])
    sy = np.sum(frame[:,1])
    n = frame.shape[0]
    num = n * sxy - sx * sy
    sx2 = np.dot(frame[:,0], frame[:,0].T)
    den = frame.shape[0]*sx2 - sx*sx
    slope = num / den
    intercept = (sy - slope * sx) / (1. * n)
    return slope, intercept

def compute_lin_reg_notime(frame):
    '''
    :param frame: np array with col1 = time, col2 = feature values
    :return: slope and intercept terms
    '''
    n = frame.shape[0]
    # temps = np.linspace(0, 1, n)
    temps = np.linspace(1, n, n)
    sxy=np.dot(temps, frame[:,1].T)
    sx = np.sum(temps)
    sy = np.sum(frame[:,1])

    num = n * sxy - sx * sy
    sx2 = np.dot(temps, temps.T)
    den = n*sx2 - sx*sx
    slope = num / den
    intercept = (sy - slope * sx) / (1. * n)
    return slope, intercept

annotator = '/home/thomas/software/sonic-annotator-1.1/sonic-annotator'
plugin='vamp:mtg-melodia:melodia:melody'
feature_dir='/baie/corpus/emoMusic/train/vamp_features'
DATADIR = '/baie/corpus/emoMusic/train/'

data = dict()
arousal, valence, song_ids, nb_of_songs = load_y_to_dict(DATADIR)

wts = np.ones(47)*1./48
wts = np.hstack((np.array([1./96]), wts, np.array([1./96])))
delay = (wts.shape[0]-1) / 2

# wts_slopes = np.array([1/4., 1/2., 1/4.])
wts_slopes = np.ones(9) * 1/8.
wts_slopes[4] = 1/4.
delay_slopes = (wts_slopes.shape[0]-1) / 2
# delay_slopes = 0

# song_id=250
# for song_id in [250]:
for song_id in song_ids:
    audio_file='/baie/corpus/emoMusic/train/audio/%d.mp3'%(song_id)

    # cline = [annotator, '-d', plugin, audio_file, '-w', 'csv', '--csv-stdout']
    # proc = Popen(cline, stdout=PIPE, stderr=PIPE)
    # feat, stdout_messages = proc.communicate()

    cline = [annotator, '-d', plugin, audio_file, '-w', 'csv', '--csv-basedir', feature_dir]
    proc = Popen(cline, stdout=PIPE, stderr=PIPE)
    _, stdout_messages = proc.communicate()

    # file='/home/thomas/software/sonic-annotator-1.1/250_vamp_mtg-melodia_melodia_melody.csv'
    file = feature_dir + '/%d_vamp_mtg-melodia_melodia_melody.csv'%(song_id)

    mel = list()
    with open(file, 'r') as csvfile:
        melreader = csv.reader(csvfile)
        for row in melreader:
            mel.append([row[0], row[1]])

    mel = np.array(mel, dtype=float)
    mel = np.abs(mel)
    # print mel.shape

    # moving average filter
    # wts = np.ones(23)*1./24
    # wts = np.hstack((np.array([1./48]), wts, np.array([1./48])))

    mel_f = np.convolve(mel[:,1], wts, mode='same')
    # take the filter delay into account:
    mel_f = np.vstack((mel[delay:,0],mel_f[delay:]))
    mel_f = mel_f.T
    mel_f_start15s = mel_f[mel_f[:,0]>15.]
    mel_start15s = mel[mel[:,0]>15.]

    # print mel_f_start15s
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(mel[:,1])
    # plt.subplot(2,1,2)
    # plt.plot(mel_f[:,1])
    # plt.show()

    # np.save('%d_vamp_mtg-melodia_melodia_melody.npy'%(song_id), mel_f)

    slopes = list()
    # start_time = 2 * 15 - delay_slopes
    start_time = 30
    ind_frame = 1
    for t in xrange(start_time,90,1):
        deb = (t -1 )/ 2.0
        fin = (t + 1) / 2.0
        mask = (mel_f[:,0]>deb) & (mel_f[:,0]<fin)
        frame = mel_f[mask,:]
        # a, b = compute_lin_reg(frame)
        a, b = compute_lin_reg_notime(frame)
        # print deb, fin, a
        # print frame[:,0]
        # print deb, a, b
        slopes.append(a)
        # if(len(slopes)>59):
        # break


    slopes = np.array(slopes)

    slopes_f = np.convolve(slopes, wts_slopes, mode='same')
    # slopes_f2 = slopes_f[delay_slopes-2:]

    # print slopes.shape, slopes_f.shape
    # a, b = compute_lin_reg(frame)
    # y = map(lambda x: a*x+b, frame[:,0])
    # plt.plot(frame[:,0], y, '-r')
    # plt.plot(frame[:,0], frame[:,1], '-r')

    data['%d'%song_id] = dict()
    data['%d'%song_id]['melody'] = np.array(slopes_f, dtype=float)
    val = np.array(valence['%d'%song_id])
    ar = np.array(arousal['%d'%song_id])

    corr_val = pearsonr(val, slopes)
    corr_val = corr_val[0]
    corr_ar = pearsonr(ar, slopes)
    corr_ar = corr_ar[0]

    corr_val_f = pearsonr(val, slopes_f)
    corr_val_f = corr_val_f[0]
    corr_ar_f = pearsonr(ar, slopes_f)
    corr_ar_f = corr_ar_f[0]
    print 'song_id %d corr (val, ar) : %.3g %.3g smoothed: : %.3g %.3g'%(song_id, corr_val, corr_ar, corr_val_f, corr_ar_f)

    # doPlot = False
    # if doPlot:
    #     # plot val, ar, slopes and smoothed slopes for a song, here the 250th song
    #     # val * 1000
    #     fig, ax = plt.subplots(2,1)
    #
    #     # ax[0].plot(1000.*val, label='valence (x10^3)')
    #     # ax[0].plot(1000.*ar, label='arousal (x10^3)')
    #     ax[0].plot(val, label='valence')
    #     ax[0].plot(ar, label='arousal')
    #     # ax[0].plot(slopes[delay_slopes:], label = 'mel slopes r_val=%.3f, r_ar=%.3f'%(corr_val, corr_ar))
    #     ax[0].plot(slopes_f, label = 'mel slopes r_val=%.3f, r_ar=%.3f'%(corr_val_f, corr_ar_f))
    #     # ax[0].plot(slopes_f2, label = 'mel slopes delayed')
    #     legend = ax[0].legend(loc='upper left', shadow=True, fontsize=11)
    #     # legend= ax[0].legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    #     # ax[1].plot((mel_start15s[:,0]-15.)*2., mel_start15s[:,1], label = 'melody')
    #     ax[1].plot((mel_f_start15s[:,0]-15.)*2., mel_f_start15s[:,1], label = 'smoothed melody')
    #     # ax[1].plot(1000.*(val+1), label='valence+1 (x10^3)')
    #     # ax[1].plot(1000.*(ar+1), label='arousal+1 (x10^3)')
    #     # ax[1].plot(slopes[delay_slopes:], label = 'mel slopes r_val=%.3f, r_ar=%.3f'%(corr_val, corr_ar))
    #     # ax[1].plot(slopes_f, label = 'mel slopes smoothed r_val=%.3f, r_ar=%.3f'%(corr_val_f, corr_ar_f))
    #     # legend = ax[1].legend(loc='upper center', shadow=True, fontsize=11)
    #     legend = ax[1].legend(loc='upper center', shadow=True, fontsize=11)
    #     plt.xlim([0,60])
    #     # fig.suptitle('song %d'%(song_id), fontsize=30)
    #     plt.show()
    #     # plt.savefig('val_ar_melody_song%d_smooth.png'%song_id)

