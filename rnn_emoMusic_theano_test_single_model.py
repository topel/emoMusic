__author__ = 'thomas'

import cPickle as pickle
import rnn_model

if __name__ == '__main__':

    # load the test dataset
    test_file = '/baie/corpus/emoMusic/test/pkl/test_set_baseline_260features_58songs_normed.pkl'
    test_data = pickle.load( open( test_file, 'rb' ) )
    X_test = test_data['test']['X']
    nb_seq_test = len(X_test)

    # load baseline RNN model
    model_file = 'RNN_models/rnn1_baseline_260feat_nh10_ne50_lr0.001_reg0.01/model_baseline_260features_431songs_normed.pkl'
    model = rnn_model.MetaRNN()
    model.load( model_file )


    pred = list()
    for ind_seq_test in xrange(nb_seq_test):
        pred.append(model.predict(X_test[ind_seq_test]))
