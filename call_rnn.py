__author__ = 'thomas'

import rnn_emoMusic_theano_cv
import numpy as np

n_hiddens = [10, 20, 30, 50, 75, 100]
n_epochs = 100
lrs = np.linspace(1e-5, 1e-2, 20)
reg_coef = 0.01

perf_file_name = 'rnn_tune_performance.log'
log_f = open(perf_file_name, 'w')

for n_hidden in n_hiddens:
    for lr in lrs:
        print 'n_hidden=%d lr=%g'%(n_hidden, lr)

        if lr < 1e-3:
            n_epochs = 2
        RMSE, pcorr = rnn_emoMusic_theano_cv.rnn_cv(n_hidden, n_epochs, lr, reg_coef)
        s = (
            'n_hidden=%d n_epochs=%d lr=%g reg_coef=%g valence: %.4f %.4f arousal: %.4f %.4f\n'
          % (n_hidden, n_epochs, lr, reg_coef, RMSE[0], pcorr[0][0], RMSE[1], pcorr[1][0])
        )
        log_f.write(s)

log_f.close()
