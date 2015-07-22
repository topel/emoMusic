__author__ = 'thomas'

import csv
import numpy as np
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE


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

song_id=250
annotator = '/home/thomas/software/sonic-annotator-1.1/sonic-annotator'
plugin='vamp:mtg-melodia:melodia:melody'
audio_file='/baie/corpus/emoMusic/train/audio/%d.mp3'%(song_id)
feature_dir='/baie/corpus/emoMusic/train/vamp_features'

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
wts = np.ones(23)*1./24
wts = np.hstack((np.array([1./48]), wts, np.array([1./48])))

mel_f = np.convolve(mel[:,1], wts, mode='same')


mel_f = np.vstack((mel[:,0],mel_f))
mel_f = mel_f.T

# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(mel[:,1])
# plt.subplot(2,1,2)
# plt.plot(mel_f[:,1])
# plt.show()

# np.save('%d_vamp_mtg-melodia_melodia_melody.npy'%(song_id), mel_f)
slopes = list()
for t in xrange(30,90,1):
    deb = t / 2.0
    fin = (t + 1) / 2.0
    mask = (mel_f[:,0]>deb) & (mel_f[:,0]<fin)
    frame = mel_f[mask,:]
    a, b = compute_lin_reg(frame)
    print deb, a, b
    slopes.append(a)

# a, b = compute_lin_reg(frame)
# y = map(lambda x: a*x+b, frame[:,0])
# plt.plot(frame[:,0], y, '-r')
# plt.plot(frame[:,0], frame[:,1], '-r')


