__author__ = 'thomas'

import numpy as np
from utils import load_X, load_y, mix, standardize, add_intercept, evaluate, evaluate1d
import matplotlib.pyplot as plt
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

srng = RandomStreams()

PURCENT = 5 # Purcentage of the set you want on the test set
NUM_FRAMES = 60
DATADIR = '/baie/corpus/emoMusic/train/'
# DATADIR = './train/'

do_regularize = False

y_, song_id, nb_of_songs = load_y(DATADIR)
X_ = load_X(DATADIR, song_id)

# Now  let's mix everything so that we can take test_set and train_set independantly
# We need to separate PER SONG
X_train, y_train, X_test, y_test, song_id_tst = mix(X_, y_, PURCENT, NUM_FRAMES, song_id, nb_of_songs)
print X_train.shape, y_train.shape, X_test.shape, y_test.shape

# standardize data
X_train, scaler = standardize(X_train)
X_test, _ = standardize(X_test, scaler)

# one dimension at a time
# y_train = y_train[:,0]
# y_test = y_test[:,0]

print X_train.shape, y_train.shape, X_test.shape, y_test.shape

tst_song = len(song_id_tst)

# add column of ones to data to account for the bias:
X_train = add_intercept(X_train)
print X_train.shape
# print X_train[0:10]

# Theano symbolic definitions

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

# def sgd(cost, params, lr=0.05):
#     grads = T.grad(cost=cost, wrt=params)
#     updates = []
#     for p, g in zip(params, grads):
#         updates.append([p, p - g * lr])
#     return updates

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, p_drop_hidden)
    h2 = rectify(T.dot(h, w_h2))

    h2 = dropout(h2, p_drop_hidden)
#    py_x = softmax(T.dot(h2, w_o))
#     return h, h2, py_x
    y = T.tanh(T.dot(h2, w_o))
    return h, h2, y


X = T.fmatrix()
Y = T.fmatrix()

nb_features = X_train.shape[1]
print 'nb_feat: ', nb_features
nb_hidden = 10
nb_output = 2

w_h = init_weights((nb_features, nb_hidden))
w_h2 = init_weights((nb_hidden, nb_hidden))
w_o = init_weights((nb_hidden, nb_output))

# h, h2, y = model(X, w_h, w_h2, w_o, 0.2, 0.5)
# noise_h, noise_h2, noise_y = model(X, w_h, w_h2, w_o, 0.2, 0.5)
h, h2, y = model(X, w_h, w_h2, w_o, 0., 0.)


lr = T.scalar('learning rate')
regul = T.scalar('L2 regul. coeff')

if do_regularize:
    # linear cost w regul
    # cost = T.mean(T.sqr(y - Y)) + regul * T.dot(w, w)
    # quadratic cost w regul
    cost = T.mean(T.sqr(T.dot(y - Y, (y - Y).T))) + regul * (T.dot(w_h, w_h.T) + T.dot(w_h2, w_h2.T) + T.dot(w_o, w_o.T))
else:
    # linear cost
    # cost = T.mean(T.sqr(y - Y))
    # quadratic cost
    # cost = T.mean(T.sqr(T.dot(y - Y, (y - Y).T)))
    cost = T.sqrt(T.mean(T.dot(y - Y, (y - Y).T)))


params = [w_h, w_h2, w_o]
# updates = sgd(cost, params, lr)
updates = RMSprop(cost, params, lr=0.001)

predict = theano.function(inputs=[X], outputs=y, allow_input_downcast=True)

if do_regularize:
    train = theano.function(inputs=[X, Y, lr, regul], outputs=cost, updates=updates, allow_input_downcast=True)
else:
    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True, mode='DebugMode')

predict = theano.function(inputs=[X], outputs=y, allow_input_downcast=True)

# Train model
print '... Training ...'
print ' REGULRAIZE: ', do_regularize

nb_iterations = 40
minibatch_size = 100
# clr = 1e-15
clr = 1e-1
# lr_decay = 0.9
lr_decay = 1.0
regul_coeff_start = 5e-1
regul_coeff = regul_coeff_start

hlr = list()
hcost = list()

if do_regularize:

    for i in range(nb_iterations):
        ccost = list()
        hlr.append(clr)
        # print 'i:', i, 'w:', w.get_value()
        for start, end in zip(range(0, len(X_train), minibatch_size), range(minibatch_size, len(X_train), minibatch_size)):
            c = train(X_train[start:end], y_train[start:end], clr, regul_coeff)
            ccost.append(c)
        hcost.append(np.mean(ccost))
        print '    ... it: %d cost: %g'%(i, hcost[-1])
        lr *= lr_decay
        # plt.plot(ccost)
        # plt.show()
else:
    for i in range(nb_iterations):
        ccost = list()
        hlr.append(clr)
        # print 'i:', i, 'w:', w.get_value()
        for start, end in zip(range(0, len(X_train), minibatch_size), range(minibatch_size, len(X_train), minibatch_size)):
            x_ =  X_train[start:end,:]
            y_ = y_train[start:end]
            # print x_.shape, y_.shape
            # print x_
            # print y_
            c = train(X_train[start:end,:], y_train[start:end])
            ccost.append(c)
        hcost.append(np.mean(ccost))
        print '    ... it: %d cost: %g'%(i, hcost[-1])
        lr *= lr_decay
        # plt.plot(ccost)
        # plt.show()


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
# pred = list()
# for cx in X_test:
#     pred.append(predict(cx))
# y_hat = np.array(pred, dtype=float)
y_hat = predict(X_test)

# RMSE, pcorr, error_per_song, mean_per_song = evaluate1d(y_test, y_hat, tst_song)
RMSE, pcorr, error_per_song, mean_per_song = evaluate(y_test, y_hat, tst_song)

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
        % (RMSE[0], RMSE[1], pcorr[0][0], pcorr[1][0])
)
