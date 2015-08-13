__author__ = 'tpellegrini'

import cPickle as pickle
import numpy as np
from utils import evaluate
from scipy import stats

def load_folds(data_dir, useEssentia=False):
    folds = list()
    for fold_id in range(10):
        print '... loading FOLD %d'%fold_id
        if useEssentia:
            fold = pickle.load( open( data_dir + '/pkl/fold%d_normed_essentia.pkl'%(fold_id), "rb" ) )

        else:
            fold = pickle.load( open( data_dir + '/pkl/fold%d_normed.pkl'%(fold_id), "rb" ) )
        folds.append(fold)

    return folds


if __name__ == '__main__':
    # DATADIR = '/baie/corpus/emoMusic/train/'
    DATADIR = './train/'

    pred_dir = 'AE/pred_gaussian_03/'
    cost_type = 'MSE'
    noise_type = 'gaussian'
    corruption_level=0.3
    n_hidden=500
    training_epochs=100

    all_fold_pred = list()

    # load reference labels
    all_fold_y_test = pickle.load ( open('ref_test_all_folds.pkl', 'r'))

    for fold_id in range(10):
        print 'fold_id: %d'%(fold_id)
        data_file = pred_dir + 'fold%d.pkl'%(fold_id)
        data = pickle.load( open( data_file, "rb" ) )
        all_fold_pred.append(data)

    all_fold_pred = [item for sublist in all_fold_pred for item in sublist]
    all_fold_pred = np.array(all_fold_pred, dtype=float)

    print all_fold_pred.shape, all_fold_y_test.shape


    # compute t-test p-values with baseline predictions
    baseline_prediction_file = 'rnn/all_baseline_predictions_260feat.pkl'
    baseline_preds = pickle.load(open( baseline_prediction_file, 'r' ))

    pvalue_val = stats.ttest_ind(baseline_preds[:,0], all_fold_pred[:,0])[1]
    pvalue_ar = stats.ttest_ind(baseline_preds[:,1], all_fold_pred[:,1])[1]
    pvalues = (pvalue_val, pvalue_ar)
    RMSE, pcorr, error_per_song, mean_per_song = evaluate(all_fold_y_test, all_fold_pred, 0)

    # print(
    #         'sklearn --> valence: %.4f, arousal: %.4f\n'
    #         'Pearson Corr --> valence: %.4f, arousal: %.4f \n'
    #         # % (RMSE[0], -1. , pcorr[0][0], -1)
    #       % (RMSE[0],RMSE[1],pcorr[0][0], pcorr[1][0])
    # )

    s = (
            'allfolds valence: %.4f %.4f arousal: %.4f %.4f p-values: %.4f, %.4f\n'
          % (RMSE[0], pcorr[0][0], RMSE[1], pcorr[1][0], pvalue_val, pvalue_ar)
    )

    print s