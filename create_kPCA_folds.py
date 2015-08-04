__author__ = 'thomas'

import numpy as np
import cPickle as pickle
import logging
import theano
from os import path, makedirs
import matplotlib.pyplot as plt
import time

from utils import evaluate, load_X_from_fold_to_3dtensor, subset_features, standardize

NUM_FRAMES = 60
NUM_OUTPUT = 2
DATADIR = '/baie/corpus/emoMusic/train/'
# DATADIR = './train/'
nb_features = 260

from sklearn.decomposition import PCA, KernelPCA
# pca = PCA(n_components=0.95, whiten='False')
pca = KernelPCA(kernel="rbf", eigen_solver = 'auto')
max_nb_samples = 10000 # number of samples to fit the kernel PCA model, if bigger then 1e+4 -> too big output dim

for fold_id in range(1,10):
    # fold_id = 0
    t0 = time.time()

    print '... loading FOLD %d'%fold_id
    fold = pickle.load( open( DATADIR + '/pkl/fold%d_normed.pkl'%(fold_id), "rb" ) )

    X_train, y_train, id_train = load_X_from_fold_to_3dtensor(fold, 'train', NUM_OUTPUT)
    X_test, y_test, id_test = load_X_from_fold_to_3dtensor(fold, 'test', NUM_OUTPUT)

    X_concat_train = np.reshape(X_train, (X_train.shape[0]*X_train.shape[1], X_train.shape[2]), order='C')
    X_concat_test = np.reshape(X_test, (X_test.shape[0]*X_test.shape[1], X_test.shape[2]), order='C')

    np.random.seed(321)
    perm = np.random.permutation(X_concat_train.shape[0])
    subset_ind = perm[0:max_nb_samples]
    X_concat_train_SUBSET = X_concat_train[subset_ind]

    start_time = time.time()
    pca_model = pca.fit(X_concat_train_SUBSET)
    print("--- kPCA fitting: %.2f seconds ---" % (time.time() - start_time))

    start_time = time.time()
    pca_X_concat_train = pca_model.transform(X_concat_train)
    print("--- kPCA transforming TRAIN: %.2f seconds ---" % (time.time() - start_time))

    start_time = time.time()
    pca_X_concat_test = pca_model.transform(X_concat_test)
    print("--- kPCA transforming TEST: %.2f seconds ---" % (time.time() - start_time))

    print 'dims: ', pca_X_concat_train.shape, pca_X_concat_test.shape
    new_dim = pca_X_concat_train.shape[1]
    X_train = np.reshape(pca_X_concat_train, (X_train.shape[0], X_train.shape[1], new_dim), order='C')
    X_test = np.reshape(pca_X_concat_test, (X_test.shape[0], X_test.shape[1], new_dim), order='C')

    data = dict()
    data['train'] = dict()
    data['train']['X'] = X_train
    data['train']['y'] = y_train
    data['train']['song_id'] = id_train
    data['test'] = dict()
    data['test']['X'] = X_test
    data['test']['y'] = y_test
    data['test']['song_id'] = id_test
    nom = DATADIR + '/pkl/fold%d_normed_kPCA10k.pkl'%(fold_id)
    pickle.dump( data, open( nom, "wb" ) )
    print ' ... output file: %s'%(nom)

