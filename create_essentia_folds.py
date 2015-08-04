__author__ = 'thomas'

import numpy as np
import cPickle as pickle
import logging
import theano
from os import path, makedirs
import matplotlib.pyplot as plt
import time
import yaml
from sklearn import preprocessing

from os import listdir
from os.path import basename, splitext


from utils import evaluate, load_X_from_fold_to_3dtensor, subset_features, standardize

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

NUM_FRAMES = 60
NUM_OUTPUT = 2
DATADIR = '/baie/corpus/emoMusic/train/'
ESSENTIA_DIR = DATADIR + 'essentia_features/'
# DATADIR = './train/'


feature_names= ['loudness',
                'spectrum_rms','spectrum_flux','spectrum_centroid','spectrum_rolloff', 'spectrum_decrease',
                'hfc','zcr',
                'mfcc','mfcc_bands',
                'barkbands', 'crest_bark', 'flatnessdb_bark', 'central_moments_bark',
                'erbbands', 'crest_erb', 'flatnessdb_erb','central_moments_erb',
                'melbands','crest_mel','flatnessdb_mel','central_moments_mel',
                'gfcc','spectral_contrast','spectral_valley','dissonance','pitchsalience','spectral_complexity',
                'danceability']

# not used: 'dynamic_complexity', first two elements of 'central_moments_bark', 'central_moments_erb', 'central_moments_mel'

# load yaml files

d_all_essentia = load_all_yaml(ESSENTIA_DIR, NUM_FRAMES)


for fold_id in range(1,10):
    # fold_id = 0
    print 'creating fold: %d'%(fold_id)
    t0 = time.time()

    print '... loading FOLD %d'%fold_id
    fold = pickle.load( open( DATADIR + '/pkl/fold%d_normed.pkl'%(fold_id), "rb" ) )

    X_train, y_train, id_train = load_X_from_fold_to_3dtensor(fold, 'train', NUM_OUTPUT)
    X_test, y_test, id_test = load_X_from_fold_to_3dtensor(fold, 'test', NUM_OUTPUT)

    essentia_X_train = list()
    essentia_X_test = list()

    for id in id_train:
        essentia_X_train.append(d_all_essentia['%d'%id])
    for id in id_test:
        essentia_X_test.append(d_all_essentia['%d'%id])

    essentia_X_train = np.array(essentia_X_train)
    essentia_X_test = np.array(essentia_X_test)

    print essentia_X_train.shape, essentia_X_test.shape
    X_concat_train = np.reshape(essentia_X_train, (essentia_X_train.shape[0]*essentia_X_train.shape[1], essentia_X_train.shape[2]), order='C')
    X_concat_test = np.reshape(essentia_X_test, (essentia_X_test.shape[0]*essentia_X_test.shape[1], essentia_X_test.shape[2]), order='C')

    print 'Normalizing...'
    print 'before: ', essentia_X_train[0,0:10]
    scaler = preprocessing.StandardScaler().fit(X_concat_train)
    X_train = scaler.transform(X_concat_train)
    X_test = scaler.transform(X_concat_test)

    new_dim = X_train.shape[1]
    essentia_X_train = np.reshape(X_train, (essentia_X_train.shape[0], essentia_X_train.shape[1], new_dim), order='C')
    essentia_X_test = np.reshape(X_test, (essentia_X_test.shape[0], essentia_X_test.shape[1], new_dim), order='C')

    print 'after: ', essentia_X_train[0,0:10]

    data = dict()
    data['train'] = dict()
    data['train']['X'] = essentia_X_train
    data['train']['y'] = y_train
    data['train']['song_id'] = id_train
    data['test'] = dict()
    data['test']['X'] = essentia_X_test
    data['test']['y'] = y_test
    data['test']['song_id'] = id_test

    data['mean'] = scaler.mean_
    data['std'] = scaler.std_

    nom = DATADIR + '/pkl/fold%d_normed_essentia.pkl'%(fold_id)
    pickle.dump( data, open( nom, "wb" ) )
    print ' ... output file: %s'%(nom)

with open('/baie/corpus/emoMusic/train/essentia_features/536.yaml', 'r') as stream:
   bob = yaml.load(stream)
feat_id=0
s = '['
for featname in feature_names:
    s += '[%d, '%(feat_id)
    tmp = np.array(bob['lowlevel'][featname])
    if len(tmp.shape) == 1:
        tmp = tmp[:, np.newaxis]
    print feat_id, featname
    feat_id += tmp.shape[1]
    if featname is 'central_moments_bark' or featname is 'central_moments_erb' or featname is 'central_moments_mel':
        feat_id -= 2
    s += '%d], '%(feat_id)
s += ']'
print s

# 0 loudness
# 1 spectrum_rms
# 2 spectrum_flux
# 3 spectrum_centroid
# 4 spectrum_rolloff
# 5 spectrum_decrease
# 6 hfc
# 7 zcr
# 8 mfcc
# 21 mfcc_bands
# 61 barkbands
# 88 crest_bark
# 89 flatnessdb_bark
# 90 central_moments_bark
# 93 erbbands
# 133 crest_erb
# 134 flatnessdb_erb
# 135 central_moments_erb
# 138 melbands
# 162 crest_mel
# 163 flatnessdb_mel
# 164 central_moments_mel
# 167 gfcc
# 180 spectral_contrast
# 186 spectral_valley
# 192 dissonance
# 193 pitchsalience
# 194 spectral_complexity
# 195 danceability

