__author__ = 'tpellegrini'

import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.stats import pearsonr


if __name__ == '__main__':
    NUM_FRAMES = 60
    DATADIR = '/baie/corpus/emoMusic/train/'
    # DATADIR = './train/'

    metadatafile = DATADIR + 'annotations/metadata.csv'
    list_genres_of_interest_file = DATADIR + 'annotations/categories.lst'
    severalGenresPerSong = True

    num_folds = 10
    fold_id = 0
    fold0 = pickle.load( open( 'train/pkl/fold%d_normed.pkl'%(fold_id), "rb" ) )

    train = fold0['train']
    test = fold0['test']

    #### plot val = f(feat) et ar = f(feat)
    cpt = 0
    for Id, dsong in train.iteritems():
        print Id
        X = dsong['X']
        val = dsong['valence']
        ar = dsong['arousal']
        plt.figure(cpt)
        plt.plot(X[:,148], val, 'x')
        plt.title('song %s'%Id)
        axes = plt.gca()
        # axes.set_xlim([-1.,1.])
        axes.set_ylim([-1.,1.])
        plt.show()
        cpt += 1
        if cpt > 5:
            break

    train_feat, val, ar = None, None, None
    for Id, dsong in train.iteritems():
        if (Id == 'mean') | (Id == 'std'):
            continue

        if train_feat is None:
            train_feat = dsong['X']
            val = dsong['valence']
            ar = dsong['arousal']
        else:
            train_feat = np.vstack((train_feat, dsong['X']))
            val = np.hstack((val, dsong['valence']))
            ar = np.hstack((ar, dsong['arousal']))

    nb_feat = train_feat.shape[1]
    s = ''
    for feat in xrange(nb_feat):
        pr_val = pearsonr(train_feat[:,feat], val)[0]
        pr_ar = pearsonr(train_feat[:,feat], ar)[0]
        if (abs(pr_val)>0.4) |  (abs(pr_ar)>0.4):
            s = s + '%d,'%feat
            print '%d, pr_val: %g, pr_ar: %g'%(feat, pr_val, pr_ar)
    print s
