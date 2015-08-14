__author__ = 'thomas'

import cPickle as pickle

if __name__ == '__main__':

    doUseEssentiaFeatures = True
    doTrainFirstRNN = True
    if doTrainFirstRNN:
        doTrainSecondRNN = False
    else:
        doTrainSecondRNN = True

    doSmoothing = True

    SUBMISSIONS_DIR = 'SUBMISSIONS/'

    if doUseEssentiaFeatures:
        nb_features = 268
    else:
        nb_features = 260

    if doTrainFirstRNN:
        MODELDIR1 = 'RNN_test/rnn1_baseline_%dfeat_nh10_ne50_lr0.001_reg0.01/'%(nb_features)
        if doSmoothing:
            predictions = MODELDIR1 + 'smoothed_predictions_test_set_baseline_%dfeatures_58songs_normed.pkl'%(nb_features)
        else:
            predictions = MODELDIR1 + 'predictions_test_set_baseline_%dfeatures_58songs_normed.pkl'%(nb_features)

        if doSmoothing:
            if doUseEssentiaFeatures:
                output_valence = SUBMISSIONS_DIR + '3/me15em_IRIT-SAMOVA_rnn%dfeatSmoothed_valence.csv'%(nb_features)
                output_arousal = SUBMISSIONS_DIR + '3/me15em_IRIT-SAMOVA_rnn%dfeatSmoothed_arousal.csv'%(nb_features)
            else:
                output_valence = SUBMISSIONS_DIR + '1/me15em_IRIT-SAMOVA_rnn%dfeatSmoothed_valence.csv'%(nb_features)
                output_arousal = SUBMISSIONS_DIR + '1/me15em_IRIT-SAMOVA_rnn%dfeatSmoothed_arousal.csv'%(nb_features)
        else:
            if doUseEssentiaFeatures:
                output_valence = SUBMISSIONS_DIR + '3/me15em_IRIT-SAMOVA_rnn%dfeat_valence.csv'%(nb_features)
                output_arousal = SUBMISSIONS_DIR + '3/me15em_IRIT-SAMOVA_rnn%dfeat_arousal.csv'%(nb_features)
            else:
                output_valence = SUBMISSIONS_DIR + '1/me15em_IRIT-SAMOVA_rnn%dfeat_valence.csv'%(nb_features)
                output_arousal = SUBMISSIONS_DIR + '1/me15em_IRIT-SAMOVA_rnn%dfeat_arousal.csv'%(nb_features)

    if doTrainSecondRNN:
        MODELDIR2 = 'RNN_test/rnn2_predictions_as_features_rnn1_baseline_%dfeat_nh10_ne50_lr0.001_reg0.01/'%(nb_features)
        predictions = MODELDIR2 + 'predictions_test_set_baseline_predictions_as_features_58songs_normed.pkl'

        if doUseEssentiaFeatures:
            output_valence = SUBMISSIONS_DIR + '4/me15em_IRIT-SAMOVA_rnn2x%dfeat_valence.csv'%(nb_features)
            output_arousal = SUBMISSIONS_DIR + '4/me15em_IRIT-SAMOVA_rnn2x%dfeat_arousal.csv'%(nb_features)
        else:
            output_valence = SUBMISSIONS_DIR + '2/me15em_IRIT-SAMOVA_rnn2x%dfeat_valence.csv'%(nb_features)
            output_arousal = SUBMISSIONS_DIR + '2/me15em_IRIT-SAMOVA_rnn2x%dfeat_arousal.csv'%(nb_features)


    pred = pickle.load( open( predictions, "rb" ) )

    print 'NB_FEATURES = %d'%(nb_features)
    print 'OUTPUTS: '
    print 'valence: ', output_valence
    print 'arousal: ', output_arousal

    max_duration = 0
    for k, v in pred.iteritems():
        if max_duration < len(v):
            max_duration = len(v)
    # print max_duration

    # print header
    header = 'song_id'
    for i in range(30, max_duration, 1):
        temps = i * 500
        # print temps
        header += ',sample_%dms'%(temps)

    with open(output_valence, 'w') as f:
        # write header
        # print header
        f.write(header + '\n')

        # write predictions
        for k, v in pred.iteritems():
            valence = k
            for i in range(30, v.shape[0]):
                valence += ',%.5f'%(v[i,0])
            f.write(valence + '\n')

    with open(output_arousal, 'w') as f:

        # write header
        f.write(header + '\n')

        # write predictions
        for k, v in pred.iteritems():
            arousal = k
            for i in range(30, v.shape[0]):
                arousal += ',%.5f'%(v[i,1])
            f.write(arousal + '\n')
