"""
 This tutorial introduces denoising auto-encoders (dA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from utils import load_X_from_fold

from sklearn import preprocessing

import gzip
import cPickle as pickle

from utils_theano import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image

def shared_dataset(data_x, data_y, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


class dA(object):
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        act_enc = 'sigmoid',
        act_dec = 'sigmoid',
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None
    ):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        assert act_enc in set(['sigmoid', 'tanh' , 'softplus' , 'rectifier'])
        assert act_dec in set(['sigmoid', 'softplus', 'linear'])
        self.act_enc = act_enc
        self.act_dec = act_dec


        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values_simple(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input_simple(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer
        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        if self.act_enc == 'sigmoid':
            return T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        elif self.act_enc == 'tanh':
            return T.tanh(T.dot(input, self.W) + self.b)
        elif self.act_enc == 'softplus':
            def softplus(x):
                return T.log(1. + T.exp(x))
            return softplus(T.dot(input, self.W) + self.b)
        elif self.act_enc == 'rectifier':
            def rectifier(x):
                return x*(x>0)
            return  rectifier(T.dot(input, self.W) + self.b)
        else:
            raise NotImplementedError('Encoder function %s is not implemented yet' \
                %(self.act_enc))


    def get_reconstructed_input(self, hidden ):
        """ Computes the reconstructed input given the values of the hidden layer """
        if self.act_dec == 'sigmoid':
            return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
        elif self.act_dec == 'linear':
            return T.dot(hidden, self.W_prime) + self.b_prime
        elif self.act_dec == 'softplus':
            def softplus(x):
                return T.log(1. + T.exp(x))
            return softplus(T.dot(hidden, self.W_prime) + self.b_prime)
        else:
            raise NotImplementedError('Decoder function %s is not implemented yet' \
                %(self.act_dec))


    def get_cost_updates_simple(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)

    def get_cost_updates(self, corruption_level, learning_rate, cost = 'CE',
        noise = 'binomial', reg = 0.0):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        # tilde_x = self.get_corrupted_input(self.x, corruption_level, noise)
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y       = self.get_hidden_values( tilde_x)
        z       = self.get_reconstructed_input(y)
        Jacobien = self.W * T.reshape(y*(1.-y),(self.n_hidden,))
        # note : we sum over the size of a datapoint; if we are using minibatches,
        #        L will  be a vector, with one entry per example in minibatch
        if cost == 'CE':
            if self.act_enc == 'tanh':
                # Moving the range of tanh from [-1;1] to [0;1]
                L = - T.sum( ((self.x+1)/2)*T.log(z) + (1-((self.x+1)/2))*T.log(1-z), axis=1 )
            else:
                L = - T.sum( self.x*T.log(z) + (1-self.x)*T.log(1-z), axis=1 )
        elif cost == 'MSE':
            L = T.sum( (self.x-z)**2, axis=1 )
        elif cost == 'jacobiCE':
            if self.act_enc == 'tanh':
                # Moving the range of tanh from [-1;1] to [0;1]
                L = - T.sum( ((self.x+1)/2)*T.log(z) + (1-((self.x+1)/2))*T.log(1-z), axis=1 )\
                    + reg * T.sum(Jacobien**2)
            else:
                L = - T.sum( self.x*T.log(z) + (1-self.x)*T.log(1-z), axis=1 )\
                    +  reg * T.sum(Jacobien**2)
        elif cost == 'jacobiMSE':
            L = T.sum((self.x - z)**2, axis=1) + reg * T.sum(Jacobien**2)
        else:
            raise NotImplementedError('This cost function %s is not implemented yet' \
                %(cost))

        # note : L is now a vector, where each element is the cross-entropy cost
        #        of the reconstruction of the corresponding example of the
        #        minibatch. We need to compute the average of all these to get
        #        the cost of the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param -  learning_rate*gparam

        return (cost, updates)

    def get_denoising_error(self, dataset, cost, noise, corruption_level):
        """ This function returns the denoising error over the dataset """
        batch_size = 100
        # compute number of minibatches for training, validation and testing
        # n_train_batches =  get_constant(dataset.shape[0]) / batch_size
        n_batches = dataset.get_value(borrow=True).shape[0] / batch_size
        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch

        cost, updates = self.get_cost_updates(corruption_level = corruption_level,
            learning_rate = 0.,
            noise = noise,
            cost = cost)
        get_error = theano.function([index], cost, updates = {}, givens = {
            self.x:dataset[index*batch_size:(index+1)*batch_size]},
            name='get_error')

        denoising_error = []
        # go through the dataset
        for batch_index in xrange(n_batches):
            denoising_error.append(get_error(batch_index))

        return numpy.mean(denoising_error)


def test_dA(learning_rate=0.1, training_epochs=100,
            batch_size=60, output_folder='dA_plots/'):

    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    cost_type = 'MSE'
    noise_type = 'gaussian'
    corruption_level=0.3
    n_hidden=500

    NUM_FRAMES = 60
    # DATADIR = '/baie/corpus/emoMusic/train/'
    DATADIR = './train/'
    fold_id = 0
    for fold_id in range(0,10):
        print '... loading FOLD %d'%fold_id
        fold = pickle.load( open( DATADIR + '/pkl/fold%d_normed.pkl'%(fold_id), "rb" ) )

        X_train, y_train, id_train = load_X_from_fold(fold, 'train')
        # scale data to [0, 1]
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        # X_train = X_train[:, 0:256]
        train_set_x, train_set_y = shared_dataset(X_train, y_train)

        X_test, y_test, id_test = load_X_from_fold(fold, 'test')
        X_test = min_max_scaler.transform(X_test)
        # X_test = X_test[:, 0:256]
        print X_test.shape
        test_set_x, test_set_y = shared_dataset(X_test, y_test)

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        print 'batch_size: %d, n_train_batches: %d'%(batch_size, n_train_batches)

        # start-snippet-2
        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch
        x = T.matrix('x')  # the data is presented as rasterized images
        # end-snippet-2

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        # os.chdir(output_folder)

        ####################################
        # BUILDING THE MODEL NO CORRUPTION #
        ####################################

        rng = numpy.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))

        da = dA(
            numpy_rng=rng,
            theano_rng=theano_rng,
            input=x,
            n_visible=260,
    #        n_visible=256,
            n_hidden=n_hidden,
            act_enc='sigmoid',
            act_dec='sigmoid'
        )

        cost, updates = da.get_cost_updates(
            corruption_level=corruption_level,
            learning_rate=learning_rate,
            cost = cost_type,
            noise = noise_type,
            # noise = 'binomial'
        )

        train_da = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size]
            }
        )

        start_time = timeit.default_timer()

        ############
        # TRAINING #
        ############

        # go through training epochs
        for epoch in xrange(training_epochs):
            # go through trainng set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(train_da(batch_index))

            print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

        end_time = timeit.default_timer()

        training_time = (end_time - start_time)

        print >> sys.stderr, ('The no corruption code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((training_time) / 60.))
        # image = Image.fromarray(
        #     tile_raster_images(X=da.W.get_value(borrow=True).T,
        #                        img_shape=(16, 16), tile_shape=(10, 10),
        #                        tile_spacing=(1, 1)))
        # image.save(output_folder + 'filters_corruption_0.png')

        # if save_dir:
        #     da.save(save_dir)

        denoising_error = da.get_denoising_error(test_set_x, cost_type,
            noise_type, corruption_level)
        print 'Training complete in %f (min) with final denoising error in test: %f' \
            %(training_time / 60.,denoising_error)

        # hidden_features = numpy(da.get_hidden_values(test_set_x))
        # print 'hidden_features: '
        # print hidden_features


        # theano functions to get representations of the dataset learned by the model
        index = T.lscalar()    # index to a [mini]batch
        x = theano.tensor.matrix('input')

        # act from the test dataset
        # need to get a T.matrix instead of a shared dataset to be able to use the get_hidden_values function
        tilde_x = da.get_corrupted_input(test_set_x, corruption_level)

        get_rep_test = theano.function([], da.get_hidden_values(x), updates = {},
            givens = {x:tilde_x},
            name = 'get_rep_test')
        test_act = get_rep_test()
        # print type(test_act)
        # print test_act.shape

        # act from the training dataset
        tilde_x = da.get_corrupted_input(train_set_x, corruption_level)
        get_rep_train = theano.function([], da.get_hidden_values(x), updates = {},
            givens = {x:tilde_x},
            name = 'get_rep_test')
        train_act = get_rep_train()

        output = dict()
        output['train'] = dict()
        output['train']['X'] = train_act
        output['train']['y'] = y_train
        output['train']['song_id'] = id_train
        output['test'] = dict()
        output['test']['X'] = test_act
        output['test']['y'] = y_test
        output['test']['song_id'] = id_test

        act_dir = 'AE/activations/'
        nom = act_dir + 'fold%d_cost%s_noise%s_level%.1f_nh%d_it%d.pkl'%(fold_id, cost_type, noise_type, corruption_level, n_hidden, training_epochs)
        pickle.dump( output, open( nom, "wb" ) )
        print 'activation (dict) saved in ' + nom

        # # start-snippet-3
        # #####################################
        # # BUILDING THE MODEL CORRUPTION 30% #
        # #####################################
        #
        # rng = numpy.random.RandomState(123)
        # theano_rng = RandomStreams(rng.randint(2 ** 30))
        #
        # da = dA(
        #     numpy_rng=rng,
        #     theano_rng=theano_rng,
        #     input=x,
        #     n_visible=16 * 16,
        #     n_hidden=500,
        #     act_enc='sigmoid',
        #     act_dec='sigmoid'
        # )
        #
        # cost, updates = da.get_cost_updates(
        #     corruption_level=corruption_level,
        #     cost = cost_type,
        #     # noise = 'binomial',
        #     noise = noise_type,
        #     learning_rate=learning_rate
        # )
        #
        # train_da = theano.function(
        #     [index],
        #     cost,
        #     updates=updates,
        #     givens={
        #         x: train_set_x[index * batch_size: (index + 1) * batch_size]
        #     }
        # )
        #
        # start_time = timeit.default_timer()
        #
        # ############
        # # TRAINING #
        # ############
        #
        # # go through training epochs
        # for epoch in xrange(training_epochs):
        #     # go through trainng set
        #     c = []
        #     for batch_index in xrange(n_train_batches):
        #         c.append(train_da(batch_index))
        #
        #     print 'Training epoch %d, cost ' % epoch, numpy.mean(c)
        #
        # end_time = timeit.default_timer()
        #
        # training_time = (end_time - start_time)
        #
        # print >> sys.stderr, ('The 30% corruption code for file ' +
        #                       os.path.split(__file__)[1] +
        #                       ' ran for %.2fm' % (training_time / 60.))
        # # end-snippet-3
        #
        # # start-snippet-4
        # image = Image.fromarray(tile_raster_images(
        #     X=da.W.get_value(borrow=True).T,
        #     img_shape=(16, 16), tile_shape=(10, 10),
        #     tile_spacing=(1, 1)))
        # image.save('filters_corruption_30.png')
        # # end-snippet-4
        #
        # # if save_dir:
        # #     da.save(save_dir)
        #
        # denoising_error = da.get_denoising_error(test_set_x, cost_type,
        #     noise_type, corruption_level)
        # print 'Training complete in %f (min) with final denoising error %f' \
        #     %(training_time / 60. ,denoising_error)

        # os.chdir('../')


if __name__ == '__main__':
    test_dA()
