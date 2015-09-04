'''
LSTM-RNN emo-music
'''
from collections import OrderedDict
import cPickle as pkl
import sys
import time
import pickle

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from utils import load_X_from_fold_to_3dtensor, evaluate1d

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    DIM_OUTPUT=2
    params = OrderedDict()
################ NOT IN OUR CASE SINCE DATA ALREADY DIGITS  ##################   
#     embedding --> to put the input at the good dimensions
#    randn = numpy.random.rand(options['n_words'],
#                              options['dim_proj'])
    
    
#    randn = numpy.random.rand(options['dim_proj'],1)
#    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    
################################################################################    
    
    params = get_layer(options['encoder'])[0](options,
                                              params,
                   

                           prefix=options['encoder'])
    # classifier
#    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
#                                            options['ydim']).astype(config.floatX)
#    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

# For a linear regression
#    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],DIM_OUTPUT).astype(config.floatX)
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],).astype(config.floatX)

    
#    params['b'] = numpy.zeros((1,)).astype(config.floatX)
    
    
    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
#breakpoint
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim, sigma=0.01):
    W = sigma*numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)

def mse(pred, y):
    # error between output and target
#    return tensor.mean((pred - y) ** 2)
    return tensor.pow(tensor.mean(tensor.pow(pred-y,2)),0.5)

def mse_numpy(pred, y, axis=None):
    # error between output and target
#    return tensor.mean((pred - y) ** 2)
    return numpy.mean((pred-y)**2/2,axis=axis)
    
def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
#    nsteps = 1
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

#    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

#    def _step(m_, x_, h_, c_):
#        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
#        preact += x_
#
#        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
#        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
#        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
#        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))
#
#        c = f * c_ + i * c
#        c = m_[:, None] * c + (1. - m_)[:, None] * c_
#
#        h = o * tensor.tanh(c)
#        h = m_[:, None] * h + (1. - m_)[:, None] * h_
#
#        return h, c

    def _step(x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
#        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
#        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
#    rval, updates = theano.scan(_step,
#                                sequences=[mask, state_below],
#                                outputs_info=[tensor.alloc(numpy_floatX(0.),
#                                                           n_samples,
#                                                           dim_proj),
#                                              tensor.alloc(numpy_floatX(0.),
#                                                           n_samples,
#                                                           dim_proj)],
#                                name=_p(prefix, '_layers'),
#                                n_steps=nsteps)
    rval, updates = theano.scan(_step,
                                sequences=[state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)

#    rval_printed = theano.printing.Print('this is a very important value')(rval[0])
#    f = theano.function([rval[0]], rval[0] * 5)
#    f_with_print = theano.function([rval[0]], rval_printed * 5)
    
#    print rval[0]
    return rval[0]


# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}


def sgd(lr, tparams, grads, x, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


#def adadelta(lr, tparams, grads, x, mask, y, cost):
def adadelta(lr, tparams, grads, x, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options, output_type = 'real'):
    trng = RandomStreams(SEED)

    #  Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

#    x = tensor.matrix('x', dtype='int64')
#    mask = tensor.matrix('mask', dtype=config.floatX)
#    y = tensor.vector('y', dtype='int64')
    
#    x = tensor.matrix('x', dtype='int64')
#    x = tensor.tensor3('x', dtype='int64')
    x = tensor.tensor3('x', dtype=config.floatX)
#    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.matrix('y', dtype=config.floatX)
#    y = tensor.vector('y', dtype=config.floatX)
    
#    n_timesteps = x.shape[0]
#    n_samples = x.shape[1]
#    print n_samples, n_timesteps    
    
    
#    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
#                                                n_samples,
#                                                options['dim_proj']])
#    print emb
                                            
#    proj = get_layer(options['encoder'])[1](tparams, emb, options,
#                                            prefix=options['encoder'],
#                                            mask=mask)
                                                
                                                
#    proj = get_layer(options['encoder'])[1](tparams, emb, options,
#                                            prefix=options['encoder'])
                                                
#    x2 = theano.shared(value=x,name='x2')
    proj = get_layer(options['encoder'])[1](tparams, x, options,
                                            prefix=options['encoder'])
                                                
#    x2 = theano.shared(x,name='x')
#    proj = get_layer(options['encoder'])[1](tparams, emb, options,
#                                            prefix=options['encoder'],
#                                            mask=mask)
                                                
## To keep the scores for every steps (and not just a mean of all scores)
#    if options['encoder'] == 'lstm':
#        proj = (proj * mask[:, :, None]).sum(axis=0)
#        proj = proj / mask.sum(axis=0)[:, None]
        
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)
    
#    pred = tensor.dot(proj, tparams['U']) + tparams['b']
    pred = tensor.dot(proj, tparams['U'])
#    pred = tensor.dot(tparams['U'],proj)
#    T.dot(x,m)+c
#    cost = T.sum(T.pow(prediction-y,2))/(2*num_samples)
    
    cost = mse(pred,y)
#    print 'DEBUG : cost is ' %(cost.shape)
#    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])
    
#    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])
   
#    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred_prob = theano.function([x], pred, name='f_pred_prob')
    
# f_pred doesn't have any sense since it's not a classification task
#    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')
#    f_pred = 0
#    off = 1e-8
#    if pred.dtype == 'float16':
#        off = 1e-6
     
        
#    if output_type == 'real':
#        cost = mse(pred,y)
#        cost = lambda y: mse(pred,y)
    #    cost = lambda y: mse(f_pred_prob,y)

#    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

#    return use_noise, x, mask, y, f_pred_prob, f_pred, cost
#    return use_noise, x, mask, y, f_pred_prob, cost
    return use_noise, x, y, f_pred_prob, cost

#def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
#    """ If you want to use a trained model, this is useful to compute
#    the probabilities of new examples.
#    """
#    n_samples = len(data[0])
#    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)
#
#    n_done = 0
#
#    for _, valid_index in iterator:
#        x, mask, y = prepare_data([data[0][t] for t in valid_index],
#                                  numpy.array(data[1])[valid_index],
#                                  maxlen=None)
#        pred_probs = f_pred_prob(x, mask)
#        probs[valid_index, :] = pred_probs
#
#        n_done += len(valid_index)
#        if verbose:
#            print '%d/%d samples classified' % (n_done, n_samples)
#
#    return probs


#def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
#    """
#    Just compute the error
#    f_pred: Theano fct computing the prediction
#    prepare_data: usual prepare_data for that dataset.
#    """
#    valid_err = 0
#    for _, valid_index in iterator:
#        x, mask, y = prepare_data([data[0][t] for t in valid_index],
#                                  numpy.array(data[1])[valid_index],
#                                  maxlen=None)
#        preds = f_pred(x, mask)
#        targets = numpy.array(data[1])[valid_index]
#        valid_err += (preds == targets).sum()
#    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])
#
#    return valid_err


#def pred_loss(f_pred_prob, prepare_data, data, iterator, verbose=False):
#    """
#    Just compute the error
#    f_pred: Theano fct computing the prediction
#    prepare_data: usual prepare_data for that dataset.
#    """
#    valid_err = 0
##    for _, valid_index in iterator:
##        x, mask, y = prepare_data([data[0][t] for t in valid_index],
##                                  numpy.array(data[1])[valid_index],
##                                  maxlen=None)
##        preds = f_pred_prob(x, mask)
##        targets = numpy.array(data[1])[valid_index]
##        targets = data[1][valid_index]
##        valid_err = mse(preds, targets)
#    
#    # En dur ici    
##    mask = numpy.ones(1,260).astype(theano.config.floatX)
#    
##    preds = f_pred_prob(data[0][iterator], mask)
#    preds = f_pred_prob(data[0][iterator])
#
##        targets = numpy.array(data[1])[valid_index]
#    targets = data[1][iterator]
#    valid_err = mse(preds, targets)
#
#    return valid_err

def pred_loss(f_pred_prob, data, iterator, verbose=False, pearsonr=None):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
#    for _, valid_index in iterator:
#        x, mask, y = prepare_data([data[0][t] for t in valid_index],
#                                  numpy.array(data[1])[valid_index],
#                                  maxlen=None)
#        preds = f_pred_prob(x, mask)
#        targets = numpy.array(data[1])[valid_index]
#        targets = data[1][valid_index]
#        valid_err = mse(preds, targets)
    
    # En dur ici    
#    mask = numpy.ones(1,260).astype(theano.config.floatX)
    
#    preds = f_pred_prob(data[0][iterator], mask)
#    preds = f_pred_prob(data[0][iterator])
#
##        targets = numpy.array(data[1])[valid_index]
#    targets = data[1][iterator]
#    valid_err = mse(preds, targets)
    
    x = numpy.array([data[0][0]])
    targets = numpy.array([data[1][0]])
    
    for _,idx in iterator:
        x = numpy.concatenate((x,data[0][idx]),axis=0)
        targets = numpy.concatenate((targets,data[1][idx]),axis=0)
        
#    x = x[1::,:,:].astype('int64')
    x = x[1::,:,:].astype(theano.config.floatX)
    
    targets = targets[1::,:].astype(theano.config.floatX)
    preds = f_pred_prob(x.transpose(1,0,2))
    
#    print x.shape
#    print targets.shape    
    # transpose car opn a transposer avant
    y_hat = preds.flatten()
    y_real = targets.transpose(1,0).flatten()
    RMSE = mean_squared_error(y_real, y_hat)**0.5
    if pearsonr != None:
        corr = pearsonr(y_real,y_hat)
        RMSE = (RMSE,corr[0])
#    RMSE, pcorr, error_per_song, mean_per_song = evaluate1d(y_hat,y_real,preds.shape[1])
    
#    valid_err = mse_numpy(preds, targets.transpose(1,0))
#    return valid_err
    return RMSE    

def train_lstm(
    dim_proj=260,  # word embeding dimension and LSTM number of hidden units.
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    n_words=10000,  # Vocabulary size
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model.npz',  # The best model will be saved there
    validFreq=370,  # Compute the validation error after this number of update.
    saveFreq=1110,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=16,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    fold_id=1,
    n_ex = None,
    dim_output=0,

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
):

    # Model options
    model_options = locals().copy()
    print "LSTM EMO --> model options", model_options

    print 'Loading data'
#    train, valid, test = load_data(n_words=n_words, valid_portion=0.05,
#                                   maxlen=maxlen)
    best_ep = 0
    NUM_OUTPUT = 2
    # A CHANGER SI SUR MON MAC
    DATADIR = '/baie/corpus/emoMusic/train/'
#    DATADIR = '..'
    print '... loading FOLD %d'%fold_id
    fold = pickle.load( open( DATADIR + '/pkl/fold%d_normed.pkl'%(fold_id), "rb" ) )

    X_train, y_train, id_train = load_X_from_fold_to_3dtensor(fold, 'train', NUM_OUTPUT)
    X_test, y_test, id_test = load_X_from_fold_to_3dtensor(fold, 'test', NUM_OUTPUT)
                                 
#    train, valid, test = (X_train, y_train) , (X_test, y_test) , (X_test, y_test)
    if n_ex == None:
                                 
        ## test only on one dimension 0 = Valence 1 = Arousal
        train, valid, test = (X_train, y_train[:,:,dim_output]) , (X_test, y_test[:,:,dim_output]) , (X_test, y_test[:,:,dim_output])  
    else:
        
    ##  test only 30 values
 ## test only on one dimension                                 
        train, valid, test = (X_train[0:n_ex,:,:], y_train[0:n_ex,:,0]) , (X_test[:,:,:], y_test[:,:,0]) , (X_test[:,:,:], y_test[:,:,0])
         ## test both dimension
#        train, valid, test = (X_train[0:n_ex,:,:], y_train[0:n_ex,:,:]) , (X_test[:,:,:], y_test[:,:,:]) , (X_test[:,:,:], y_test[:,:,:])
              
##################### A VOIR #############################################
#    if test_size > 0:
#        # The test set is sorted by size, but we want to keep random
#        # size example.  So we must select a random selection of the
#        # examples.
#        idx = numpy.arange(len(test[0]))
#        numpy.random.shuffle(idx)
#        idx = idx[:test_size]
#        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])
###########################################################################

##################### NO YDIM SINCE LINEAR REGRESSION ########################
#    ydim = numpy.max(train[1]) + 1
#
#    model_options['ydim'] = ydim

    print 'Building model'
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('lstm_model.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    

    # use_noise is for dropout
#    (use_noise, x, mask,
#     y, f_pred_prob, cost) = build_model(tparams, model_options)
    (use_noise, x,
     y, f_pred_prob, cost) = build_model(tparams, model_options)
#    x0 = tensor.matrix('x0', dtype=config.floatX)
#    (use_noise, x, mask,
#     y, f_pred_prob, f_pred, cost) = build_model(x0, tparams, model_options)
     
    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

#    f_cost = theano.function([x, mask, y], cost, name='f_cost')
    f_cost = theano.function([x, y], cost, name='f_cost')
    
    grads = tensor.grad(cost, wrt=tparams.values())
#    f_grad = theano.function([x, mask, y], grads, name='f_grad')
    f_grad = theano.function([x, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
#    f_grad_shared, f_update = optimizer(lr, tparams, grads,
#                                        x, mask, y, cost)
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, y, cost)
    print 'Optimization'
    
    # shuffle index for the test and validation sets 
    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print "%d train examples" % len(train[0])
    print "%d valid examples" % len(valid[0])
    print "%d test examples" % len(test[0])

    history_errs = []
    tot_cost = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in xrange(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
#                use_noise.set_value(1.)
                use_noise.set_value(0.)
                
                # Select the random examples for this minibatch
#                y = [train[1][t] for t in train_index]
#                x = [train[0][t]for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)

#                x, mask, y = prepare_data(x, y)
#                mask = numpy.ones((dim_proj,1)).astype(theano.config.floatX)
#                mask = numpy.ones((1,dim_proj)).astype(theano.config.floatX)
                               
#                x = train[0][train_index].astype('int64')
                x = train[0][train_index].transpose(1,0,2).astype(config.floatX)
                
#                x = numpy.array([train[0][t,:,:] for t in train_index]
#                x = X_train 
                y = train[1][train_index].astype(theano.config.floatX)
#                y = train[1][train_index,:].astype(theano.config.floatX)
#                y = y_train[train_index].astype(theano.config.floatX)
#                cost = f_grad_shared(x, mask, y, allow_input_downcast=True)
                
                n_samples += x.shape[1]
                
#                print x.shape
#                cost = f_grad_shared(x, mask, y)
#                theano.printing.debugprint(f_grad_shared)
                
                cost = f_grad_shared(x.transpose(1,0,2), y)
                tot_cost.append(cost)
                
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    
#                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost
                    print 'Epoch ', eidx, 'Update ', uidx, 'tot_Cost ', numpy.mean(tot_cost)

                    
                    tot_cost = []
                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print 'Saving...',

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print 'Done'

                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
#                    train_err = pred_error(f_pred, prepare_data, train, kf)
#                    valid_err = pred_error(f_pred, prepare_data, valid,
#                                           kf_valid)
#                    test_err = pred_error(f_pred, prepare_data, test, kf_test)

#                    train_err = cost
                    train_err = pred_loss(f_pred_prob, train,
                                           kf)
#                    f_pred_prob([])
                    valid_err = pred_loss(f_pred_prob, valid,
                                           kf_valid)
                    test_err = pred_loss(f_pred_prob, test, kf_test)
                    
                    history_errs.append([valid_err, test_err])

                    if (uidx == 0 or
                        valid_err <= numpy.array(history_errs)[:,
                                                               0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0
                        best_ep = eidx

#                    print ('Train ', train_err.flatten()[0], 'Valid ', valid_err,
#                           'Test ', test_err)
                    print ('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err)
                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

            print 'Seen %d samples' % n_samples

            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:#
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
#    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
#    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
#    test_err = pred_error(f_pred, prepare_data, test, kf_test)
    
    train_err, tr_corr = pred_loss(f_pred_prob, train, kf_train_sorted, pearsonr=pearsonr)
    valid_err, val_corr = pred_loss(f_pred_prob, valid, kf_valid, pearsonr=pearsonr)
    test_err = pred_loss(f_pred_prob, test, kf_test)

    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))
    return train_err, valid_err, test_err, tr_corr, val_corr, best_ep


if __name__ == '__main__':
    
 # Number of example in the train set : for debugging only   
 ex = None
 batch_size = 16  
 if ex == None and batch_size == 16:
     validFreq = 50
 elif ex == 25 and batch_size == 1:
     validFreq = ex
 else:
     validFreq = 10
     
# validFreq = 1    
 n_fold=10
 max_epochs = 100
 patience=15  # Number of epoch to wait before early stop if no progress
 dispFreq=validFreq  # Display to stdout the training progress every N updates
 
 # 0 for the Valence, and 1 for the Arousal !! 
 dim_output = 0 


 train_err_tot = []
 valid_err_tot = []
 tr_corr_tot = []
 val_corr_tot = [] 
 best_ep_tot = []
 
 
 for fold_id in range(n_fold):    
    # See function train for all possible parameter and there definition.
    theano.config.compute_test_value = 'off'
    train_err, valid_err, _, tr_corr, val_corr, best_ep = train_lstm(
        
        max_epochs=max_epochs,
        validFreq=validFreq,
        patience=patience,  # Number of epoch to wait before early stop if no progress
        dispFreq=dispFreq,  # Display to stdout the training progress every N updates
        n_ex = ex, # number of example 
        batch_size=batch_size,  # The batch size during training.
        dim_output=dim_output, # for Valence or Arousal
                 
        test_size=500,
        dim_proj=260,
        optimizer=adadelta,
        n_words=10000,  # Vocabulary size
        lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
        decay_c=0.9,  # Weight decay for the classifier applied to the U weights.
        
#        n_words=250000,


        encoder='lstm',  # TODO: can be removed must be lstm.
        saveto='lstm_model.npz',  # The best model will be saved there
        saveFreq=1110,  # Save the parameters after every saveFreq updates
        maxlen=100,  # Sequence longer then this get ignored

        valid_batch_size=64,  # The batch size used for validation/test set.
        fold_id=fold_id,
        )
    train_err_tot.append(train_err)
    valid_err_tot.append(valid_err)
    tr_corr_tot.append(tr_corr)
    val_corr_tot.append(val_corr)
    best_ep_tot.append(best_ep)
    
    
    
 numpy.set_printoptions(precision=3) 
 
 
 print "---------------------------TRAIN---------------------------------"
 print "train_err par fold : "
 print(numpy.array([train_err_tot[q] for q in range(len(train_err_tot))]))
 print "train_corr par fold : "
 print(numpy.array([tr_corr_tot[q] for q in range(len(tr_corr_tot))]))
 print "train_err MEAN : ", numpy.mean(train_err_tot), "train_corr MEAN : ", numpy.mean(tr_corr_tot)

 ## VALIDATION
 print "---------------------------VALIDATION---------------------------------"
 print "val_err par fold : "
 print(numpy.array([valid_err_tot[q] for q in range(len(valid_err_tot))]))
 print "train_corr par fold : "
 print(numpy.array([val_corr_tot[q] for q in range(len(val_corr_tot))]))
 print "valid_err MEAN ", numpy.mean(valid_err_tot), "valid_corr MEAN ", numpy.mean(val_corr_tot)
 print "---------------------------ITERATION---------------------------------"
 print "Epoque meilleure valid", 
 print(numpy.array([best_ep_tot[q] for q in range(len(best_ep_tot))]))
 print "Moyenne meilleure epoque valid", numpy.mean(best_ep_tot)
