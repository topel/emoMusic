__author__ = 'thomas'

import numpy as np
from utils import load_X, load_y, mix, standardize, add_intercept, evaluate, evaluate1d
import matplotlib.pyplot as plt
import theano
from theano import tensor as T

PURCENT = 5 # Purcentage of the set you want on the test set
NUM_FRAMES = 60
# DATADIR = '/baie/corpus/emoMusic/train/'
DATADIR = './train/'

do_regularize = False

y_, song_id, nb_of_songs = load_y(DATADIR)
X_ = load_X(DATADIR, song_id)

# Now  let's mix everything so that we can take test_set and train_set independantly
# We need to separate PER SONG
X_train, y_train, X_test, y_test, song_id_tst = mix(X_, y_, PURCENT, NUM_FRAMES, song_id, nb_of_songs)
print X_train.shape, y_train.shape, X_test.shape, y_test.shape
# print X_train[0:3,0:3]
# print np.mean(X_train[:,0:3], axis=0), np.std(X_train[:,0:3], axis=0)
# print np.mean(X_test[:,0:3], axis=0), np.std(X_test[:,0:3], axis=0)

# with(open('train_dummy.txt', mode='w')) as infile:
#     for i in range(X_train.shape[0]):
#         s=''
#         for feat in range(3):
#             s = s + '%g '%X_train[i,feat]
#         infile.write('%s\n'%s)

# standardize data
X_train, scaler = standardize(X_train)
X_test, _ = standardize(X_test, scaler)

# print np.mean(X_train[:,0:3], axis=0), np.std(X_train[:,0:3], axis=0)
# print np.mean(X_test[:,0:3], axis=0), np.std(X_test[:,0:3], axis=0)

# with(open('train_dummy_normed.txt', mode='w')) as infile:
#     for i in range(X_train.shape[0]):
#         s=''
#         for feat in range(3):
#             s = s + '%g '%X_train[i,feat]
#         infile.write('%s\n'%s)

# one dimension at a time
y_train = y_train[:,0]
y_test = y_test[:,0]

print X_train.shape, y_train.shape, X_test.shape, y_test.shape

tst_song = len(song_id_tst)

# add column of ones to data to account for the bias:
X_train = add_intercept(X_train)
print X_train.shape
# print X_train[0:10]

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
    cost = T.mean(T.sqr(T.dot(y - Y, y - Y)))

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

nb_iterations = 150
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

        plt.plot(ccost)
        plt.show()

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

doplotCost = True
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
# add column of ones to data to account for the bias:
X_test = add_intercept(X_test)
print X_test.shape
pred = list()
for cx in X_test:
    pred.append(predict(cx))

y_hat = np.array(pred, dtype=float)

RMSE, pcorr, error_per_song, mean_per_song = evaluate1d(y_test, y_hat, tst_song)

All_stack =np.hstack(( error_per_song, mean_per_song ))
print'  Error per song (ar/val)  Mean_per_song (ar/val)    :\n'
print(All_stack)
print '\n'

print'song_id :'
print(song_id_tst)
print '\n'
#print('Error per song: \n', Error_per_song)

print(
        'sklearn --> arrousal : %.4f, valence : %.4f\n'
        'Pearson Corr --> arrousal : %.4f, valence : %.4f \n'
        % (RMSE[0], -1. , pcorr[0][0], -1)
      # % (RMSE[0],RMSE[1],pcorr[0][0], pcorr[1][0])
)
