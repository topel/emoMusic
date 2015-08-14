__author__ = 'thomas'

import cPickle as pickle
import rnn_model
import numpy as np

if __name__ == '__main__':

    doUseEssentiaFeatures = True
    runFirstModel = True
    runSecondModel = False
    doSmoothing = True

    if doUseEssentiaFeatures:
        nb_features = 268
    else:
        nb_features = 260

    print '... using %d features ...'%(nb_features)
    print '... smoothing preds: %s ...'%(doSmoothing)

    # smooth prediction params
    taille = 12
    wts = np.ones(taille-1)*1./taille
    wts = np.hstack((np.array([1./(2*taille)]), wts, np.array([1./(2*taille)])))
    delay = (wts.shape[0]-1) / 2

    if runFirstModel:
        # load the test dataset
        test_file = '/baie/corpus/emoMusic/test/pkl/test_set_baseline_%dfeatures_58songs_normed.pkl'%(nb_features)
        test_data = pickle.load( open( test_file, 'rb' ) )
        nb_seq_test = len(test_data)

        # load baseline RNN model
        basename = 'rnn1_baseline_%dfeat_nh10_ne50_lr0.001_reg0.01'%(nb_features)
        model_file = 'RNN_models/' + basename + '/model_baseline_%dfeatures_431songs_normed.pkl'%(nb_features)
        model = rnn_model.MetaRNN()
        model.load( model_file )

        # predict!
        pred = dict()
        for id, v in test_data.iteritems():
            X = v['X']
            print id, X.shape
            pred[id] = model.predict(X)
            if doSmoothing:
                y_hat = np.array(pred[id], dtype=float)
                y_hat_smooth = np.zeros_like(y_hat, dtype=float)
                y_hat_smooth[:, 0] = np.convolve(y_hat[:, 0], wts, mode='same')
                y_hat_smooth[:delay, 0] = y_hat[:delay, 0]
                y_hat_smooth[-delay:, 0] = y_hat[-delay:, 0]
                y_hat_smooth[:, 1] = np.convolve(y_hat[:, 1], wts, mode='same')
                y_hat_smooth[:delay, 1] = y_hat[:delay, 1]
                y_hat_smooth[-delay:, 1] = y_hat[-delay:, 1]
                pred[id] = y_hat_smooth

        # save predictions
        if doSmoothing:
            pred_file = 'RNN_test/' + basename + '/smoothed_predictions_test_set_baseline_%dfeatures_58songs_normed.pkl'%(nb_features)
        else:
            pred_file = 'RNN_test/' + basename + '/predictions_test_set_baseline_%dfeatures_58songs_normed.pkl'%(nb_features)

        pickle.dump( pred, open( pred_file, 'wb' ) )
        print ' ... --> saved to: ** %s **'%(pred_file)

    if runSecondModel:

        # load predictions from rnn1:
        basename = 'rnn1_baseline_%dfeat_nh10_ne50_lr0.001_reg0.01'%(nb_features)
        pred_file = 'RNN_test/' + basename + '/predictions_test_set_baseline_%dfeatures_58songs_normed.pkl'%(nb_features)
        first_pred = pickle.load( open( pred_file, 'rb' ))

        # load prediction_as_feature RNN2 model
        basename = 'rnn2_predictions_as_features_rnn1_baseline_%dfeat_nh10_ne50_lr0.001_reg0.01'%(nb_features)
        model_file = 'RNN_models/' + basename + '/model_baseline_predictions_as_features_431songs_normed.pkl'
        model = rnn_model.MetaRNN()
        model.load( model_file )

        # predict!
        pred2 = dict()
        for id, v in first_pred.iteritems():
            pred2[id] = model.predict(v)

        # save predictions
        pred_file2 = 'RNN_test/' + basename + '/predictions_test_set_baseline_predictions_as_features_58songs_normed.pkl'
        pickle.dump( pred2, open( pred_file2, 'wb' ) )
        print ' ... --> saved to: ** %s **'%(pred_file2)
