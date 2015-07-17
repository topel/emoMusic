__author__ = 'thomas'

# Load modules and data
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import cPickle as pickle
import theano
from theano import tensor as T

from utils import load_X, load_y, mix, standardize, add_intercept, evaluate, evaluate1d, load_X_from_fold
import matplotlib.pyplot as plt



PURCENT = 5 # Purcentage of the set you want on the test set
NUM_FRAMES = 60
DATADIR = '/baie/corpus/emoMusic/train/'
# DATADIR = './train/'


EMO='valence'
# EMO='arousal'
do_regularize = False

# fold_id = 2

all_fold_pred = list()
all_fold_y_test = list()
all_fold_id_test = list()

for fold_id in range(10):
    print '... loading FOLD %d'%fold_id
    fold = pickle.load( open( DATADIR + '/pkl/fold%d_normed.pkl'%(fold_id), "rb" ) )

    X_train, y_train, id_train = load_X_from_fold(fold, 'train')
    X_test, y_test, id_test = load_X_from_fold(fold, 'test')

    print id_test.shape

    # X_train = X_train[:,[10,12,13,17,19,82,83,84,85,89,90,91,103,140,142,146,148,212,214,218,220]]
    # X_test = X_test[:,[10,12,13,17,19,82,83,84,85,89,90,91,103,140,142,146,148,212,214,218,220]]
    # X_train = X_train[:,[13,85,103,142,214]]
    # X_test = X_test[:,[13,85,103,142,214]]

    # one dimension at a time
    # 0: arousal, 1: valence
    if EMO == 'valence':
        print '... emotion: valence'
        y_train = y_train[:,0]
        y_test = y_test[:,0]
    else:
        print '... emotion: arousal'
        y_train = y_train[:,1]
        y_test = y_test[:,1]

    # X_test = X_train[119:119+y_test.shape[0],:]
    # y_test = y_train[119:119+y_test.shape[0]]

    # add column of ones to data to account for the bias:
    X_train = add_intercept(X_train)
    X_test = add_intercept(X_test)
    print X_train.shape, y_train.shape, X_test.shape, y_test.shape

    # Theano symbolic definitions
    X = T.vector()
    Y = T.scalar()
    lr = T.scalar('learning rate')
    regul = T.scalar('L2 regul. coeff')

    def model(X, w):
        return T.dot(X, w)

    nb_features = X_train.shape[1]
    print 'nb_feat: ', nb_features
    w = theano.shared(np.zeros(nb_features, dtype=theano.config.floatX))

    y = model(X, w)

    if do_regularize:
        # linear cost w regul
        # cost = T.mean(T.sqr(y - Y)) + regul * T.dot(w, w)
        # quadratic cost w regul
        cost = T.mean(T.sqr(T.dot(y - Y, y - Y))) + regul * T.dot(w, w)
    else:
        # linear cost
        # cost = T.mean(T.sqr(y - Y))
        # quadratic cost
        # cost = T.mean(T.sqr(T.dot(y - Y, y - Y)))
        cost = T.sqrt(T.mean(T.dot(y - Y, y - Y)))

    gradient = T.grad(cost=cost, wrt=w)
    updates = [[w, w - gradient * lr]]

    if do_regularize:
        train = theano.function(inputs=[X, Y, lr, regul], outputs=cost, updates=updates, allow_input_downcast=True)
    else:
        train = theano.function(inputs=[X, Y, lr], outputs=cost, updates=updates, allow_input_downcast=True)

    predict = theano.function(inputs=[X], outputs=y, allow_input_downcast=True)

    # Train model
    print '... Training ...'
    print ' REGULRAIZE: ', do_regularize

    nb_iterations = 100
    # clr = 1e-15
    clr = 1e-6
    # lr_decay = 0.9
    lr_decay = 1.0
    # here we test several regul coeffs
    regul_coeff_start = 1e-7
    regul_coeff = regul_coeff_start
    # regul_multiplier = 5

    hlr = list()
    hcost = list()

    if do_regularize:
        for i in range(nb_iterations):
            print '... it: %d'%i
            ccost = list()
            hlr.append(clr)
            # print 'i:', i, 'w:', w.get_value()
            ind_it=1
            for cx, cy in zip(X_train, y_train):
                c = train(cx, cy, clr, regul_coeff)
                ccost.append(c)
                print cx[0:3], cy, c
                if ind_it % 100 == 0:
                    break
                ind_it += 1
            hcost.append(np.mean(ccost))
            lr *= lr_decay

    #        plt.plot(ccost)
    #        plt.show()

    else:
        for i in range(nb_iterations):
            ccost = list()
            hlr.append(clr)
            # print 'i:', i, 'w:', w.get_value()
            ind_it=1
            for cx, cy in zip(X_train, y_train):
                c = train(cx, cy, clr)
                ccost.append(c)
                # print cx[0:3], cy, c
                # if ind_it % 100 == 0:
                #     break
                ind_it += 1
            hcost.append(np.mean(ccost))
            print '    ... it: %d cost: %g'%(i, hcost[-1])

            lr *= lr_decay
            # plt.plot(ccost)
            # plt.show()

    W = w.get_value()
    # print 'train: regul=%g finalCost=%g'%(regul_coeff, hcost[-1])
    print '... finalCost=%g'%(hcost[-1])
    # print W

    doplotCost = False
    if doplotCost:
        regul_coeff = regul_coeff_start
        fig = plt.figure()

        plt.plot(hcost)
        # plt.plot(hcost[i], label='reg=%.02e cost=%.03g w1=%.3f w2=%.3f b=%.3f'%(regul_coeff, hcost[i][-1], W[i][0], W[i][1], W[i][2]))
        # ax = plt.gca()
        # legend = ax.legend(loc='upper right', shadow=True)
        # ax.legend(bbox_to_anchor=(1.1, 1.05))
        # regul_coeff *= regul_multiplier

        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()
        # fig.savefig('linReg_regularization_comparison.eps', format='eps')

    # predict and eval on test set
    print '... predicting ...'

    pred = list()
    for cx in X_test:
        pred.append(predict(cx))

    all_fold_pred.append(pred)
    all_fold_y_test.append(y_test.tolist())

    y_hat = np.array(pred, dtype=float)

    RMSE, pcorr, error_per_song, mean_per_song = evaluate1d(y_test, y_hat, id_test.shape[0])

    # All_stack =np.hstack(( error_per_song, mean_per_song ))
    # print'  Error per song (ar/val)  Mean_per_song (ar/val)    :\n'
    # print(All_stack)
    # print '\n'
    #
    # print'song_id :'
    # print(id_test)
    # print '\n'
    # #print('Error per song: \n', Error_per_song)

    print(
            'sklearn --> arrousal : %.4f, valence : %.4f\n'
            'Pearson Corr --> arrousal : %.4f, valence : %.4f \n'
            % (RMSE[0], -1. , pcorr[0][0], -1)
          # % (RMSE[0],RMSE[1],pcorr[0][0], pcorr[1][0])
    )

    doPlot = False
    if doPlot:
        fig, ax = plt.subplots()
        x1 = np.linspace(1,len(pred),len(pred))
        ax.plot(x1, y_test, 'o', label="Data")
        ax.plot(x1, y_hat, 'r-', label="OLS prediction")
        plt.title(EMO + ' on Test subset')
        ax.legend(loc="best")
        plt.show()
        # plt.savefig('figures/linreg_sm_%s_fold%d.png'%(EMO, fold_id), format='png')

all_fold_pred = [item for sublist in all_fold_pred for item in sublist]
all_fold_y_test = [item for sublist in all_fold_y_test for item in sublist]

all_fold_pred = np.array(all_fold_pred, dtype=float)
all_fold_y_test = np.array(all_fold_y_test, dtype=float)

print all_fold_pred.shape, all_fold_y_test.shape

RMSE, pcorr, error_per_song, mean_per_song = evaluate1d(all_fold_y_test, all_fold_pred, len(all_fold_id_test))

if EMO == 'valence':
    print(
            'RMSE valence : %.4f\n'
            'PearsonCorr valence : %.4f \n'
            % (RMSE[0], pcorr[0][0])
    )
else:
    print(
            'RMSE arousal : %.4f\n'
            'PearsonCorr arousal : %.4f \n'
            % (RMSE[0], pcorr[0][0])
    )

