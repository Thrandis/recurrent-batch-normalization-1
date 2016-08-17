import sys, os, util, functools
import shutil
import logging
from collections import OrderedDict
import numpy as np
import theano, theano.tensor as T
from theano import function
from theano.sandbox.rng_mrg import MRG_RandomStreams
import blocks.config
import fuel.datasets, fuel.streams, fuel.transformers, fuel.schemes

from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.algorithms import GradientDescent, RMSProp, StepClipping, CompositeRule, Momentum, Adam
from blocks.model import Model
from blocks.extensions import FinishAfter, Printing, ProgressBar, Timing, SimpleExtension
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.training import TrackTheBest
from blocks.extensions.saveload import Checkpoint
from extensions import DumpLog, DumpBest, PrintingTo, DumpVariables, SharedVariableModifier
from blocks.main_loop import MainLoop
from blocks.utils import shared_floatx_zeros
from blocks.roles import add_role, PARAMETER

logging.basicConfig()
logger = logging.getLogger(__name__)

class ForceL2Norm(SimpleExtension):

    def __init__(self, variables, **kwargs):
        kwargs.setdefault('before_first_epoch', True)
        kwargs.setdefault('after_batch', True)
        super(ForceL2Norm, self).__init__(**kwargs)
        self.variables = variables
        updates = []
        for variable in variables:
            norm = T.sqrt((variable**2).sum(axis=0, keepdims=True))  #TODO Check axis
            updates.append((variable, variable/norm))
        self.function = function([], [], updates=updates)

    def do(self, which_callback, *args):
        self.function()

def learning_rate_decayer(decay_rate, i, learning_rate):
    return ((1. - decay_rate) * learning_rate).astype(theano.config.floatX)

def zeros(shape):
    return np.zeros(shape, dtype=theano.config.floatX)

def ones(shape):
    return np.ones(shape, dtype=theano.config.floatX)

def glorot(shape):
    d = np.sqrt(6. / sum(shape))
    return np.random.uniform(-d, +d, size=shape).astype(theano.config.floatX)

def orthogonal(shape):
    # taken from https://gist.github.com/kastnerkyle/f7464d98fe8ca14f2a1a
    """ benanne lasagne ortho init (faster than qr approach)"""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v  # pick the one with the correct shape
    q = q.reshape(shape)
    return q[:shape[0], :shape[1]].astype(theano.config.floatX)

def uniform(shape, scale):
    return np.random.uniform(-scale, +scale, size=shape).astype(theano.config.floatX)

def softmax_lastaxis(x):
    # for sequence of distributions
    return T.nnet.softmax(x.reshape((-1, x.shape[-1]))).reshape(x.shape)

def crossentropy_lastaxes(yhat, y):
    # for sequence of distributions/targets
    return -(y * T.log(yhat)).sum(axis=yhat.ndim - 1)

_data_cache = dict()
def get_data(which_set):
    if which_set not in _data_cache:
        if not os.path.exists('/Tmp/laurent/data/char_level_penntree.npz'):
            if not os.path.exists('/Tmp/laurent/data'):
                os.makedirs('/Tmp/laurent/data')
            shutil.copy('/data/lisa/data/PennTreebankCorpus/char_level_penntree.npz',
                        '/Tmp/laurent/data/')
        data = np.load('/Tmp/laurent/data/char_level_penntree.npz')
        # put the entire thing on GPU in one-hot (takes
        # len(self.vocab) * len(self.data) * sizeof(floatX) bytes
        # which is about 1G for the training set and less for the
        # other sets)
        CudaNdarray = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.CudaNdarray
        # (doing it in numpy first because cudandarray doesn't accept
        # lists of indices)
        one_hot_data = np.eye(len(data["vocab"]), dtype=theano.config.floatX)[data[which_set]]
        #_data_cache[which_set] = CudaNdarray(one_hot_data)
        _data_cache[which_set] = one_hot_data
    return _data_cache[which_set]

class PTB(fuel.datasets.Dataset):
    provides_sources = ('features',)
    example_iteration_scheme = None

    def __init__(self, which_set, length, augment=False):
        self.which_set = which_set
        self.length = length
        self.augment = augment
        self.data = get_data(which_set)
        self.num_examples = int(len(self.data) / self.length)
        if self.augment:
            # -1 so we have one self.length worth of room for augmentation
            self.num_examples -= 1
        super(PTB, self).__init__()

    def open(self):
        offset = 0
        if self.augment:
            # choose an offset to get some data augmentation by not always chopping
            # the examples at the same point.
            offset = np.random.randint(self.length)
        # none of this should copy
        data = self.data[offset:]
        # reshape to nonoverlapping examples
        data = (data[:self.num_examples * self.length]
                .reshape((self.num_examples, self.length, self.data.shape[1])))
        # return the data so we will get it as the "state" argument to get_data
        return data

    def get_data(self, state, request):
        if isinstance(request, (tuple, list)):
            request = np.array(request, dtype=np.int64)
            return (state.take(request, 0),)
        return (state[request],)

def get_stream(which_set, batch_size, length, num_examples=None, augment=False):
    dataset = PTB(which_set, length=length, augment=augment)
    if num_examples is None or num_examples > dataset.num_examples:
        num_examples = dataset.num_examples
    stream = fuel.streams.DataStream.default_stream(
        dataset,
        iteration_scheme=fuel.schemes.ShuffledScheme(num_examples, batch_size))
    return stream

activations = dict(
    tanh=T.tanh,
    identity=lambda x: x,
    relu=lambda x: T.max(0, x))

class Parameters(object):
    pass

def norm_tanh(x):
    y = T.tanh(x)
    return y / T.sqrt(0.394)


def norm_tanh_sqrt(x):
    y = T.tanh(x/T.sqrt(2))
    return y / T.sqrt(0.394)

def norm_tanh_gamma(x, gamma):
    if gamma == 0.1:
        scale = T.sqrt(0.0098) 
    elif gamma == 1.0:
        scale = T.sqrt(0.3944)
    elif gamma == 0.8:
        scale = T.sqrt(0.3141)
    else:
        raise NotImplementedError
    return T.tanh(x) / scale

def norm_sigmoid(x):
    y = T.nnet.sigmoid(x)
    return (y - 0.5) / T.sqrt(0.043)


class RNN(object):
    def __init__(self, args, nclasses, norm=False):
        print 'RNN'
        self.nclasses = nclasses
        self.norm = norm
        if self.norm:
            self.tanh = norm_tanh
        else:
            self.tanh = T.tanh

    def allocate_parameters(self, args):
        if hasattr(self, "parameters"):
            return self.parameters

        self.parameters = Empty()

        h0 = theano.shared(zeros((args.num_hidden,)), name="h0")
        if args.init == "id":
            Wa = theano.shared(np.eye(args.num_hidden).astype(theano.config.floatX), name="Wa")
        else:
            Wa = theano.shared(orthogonal((args.num_hidden, args.num_hidden)), name="Wa")
        Wx = theano.shared(orthogonal((self.nclasses, args.num_hidden)), name="Wx")
        ab_betas = theano.shared(args.initial_beta  * ones((args.num_hidden,)), name="ab_betas")

        parameters_list = [Wa, Wx, h0, ab_betas]
        if self.norm:
            a_gammas = theano.shared(args.initial_gamma * ones((args.num_hidden,)), name="a_gammas")
            b_gammas = theano.shared(args.initial_gamma * ones((args.num_hidden,)), name="b_gammas")
            parameters_list.extend([a_gammas, b_gammas])
        for parameter in parameters_list:
            print parameter.name
            add_role(parameter, PARAMETER)
            setattr(self.parameters, parameter.name, parameter)
        return self.parameters

    def construct_graph(self, args, x, length):
        p = self.allocate_parameters(args)

        if self.norm:
            # Normalize Wa and Wx, and apply gammas
            norm_Wx = p.Wx / T.sqrt((p.Wx**2).sum(axis=0, keepdims=True)) # TODO Check axis
            norm_Wa = p.Wa / T.sqrt((p.Wa**2).sum(axis=0, keepdims=True)) # TODO Check axis
            norm_Wx *= p.b_gammas.dimshuffle('x', 0) # TODO Check axis
            norm_Wa *= p.a_gammas.dimshuffle('x', 0) # TODO Check axis
        else:
            norm_Wx = p.Wx
            norm_Wa = p.Wa
 
        xtilde = T.dot(x, norm_Wx)

        dummy_states = dict(h=T.zeros((xtilde.shape[0], xtilde.shape[1], args.num_hidden)))

        if self.norm:
            print 'NORM and SQRT(2) and NORMTanh' 
        def stepfn(x, dummy_h, h, norm_Wa, ab_betas):
            atilde = T.dot(h, norm_Wa)
            btilde = x
            a_normal = atilde
            b_normal = btilde
            if self.norm:
                ab = (a_normal + b_normal) / T.sqrt(2) + ab_betas
            else:
                ab = a_normal + b_normal + ab_betas
            htilde = ab
            if self.norm:            
                h = dummy_h + norm_tanh(ab)
            else:
                h = dummy_h + T.tanh(ab)
            return h, atilde, btilde, htilde

        [h, atilde, btilde, htilde], _ = theano.scan(
            stepfn,
            sequences=[xtilde, dummy_states["h"]],
            non_sequences=[norm_Wa, p.ab_betas],
            outputs_info=[T.repeat(p.h0[None, :], xtilde.shape[1], axis=0),
                          None, None, None])
        return dict(h=h, atilde=atilde, btilde=btilde, htilde=htilde), dummy_states


class Empty():
    pass


class LSTM(object):
    def __init__(self, args, nclasses, norm=False):
        self.nclasses = nclasses
        self.norm = norm
        print 'STANDARD NONLINEARITIES and 2xnorm_tanh'
        if False: #norm:
            self.tanh = norm_tanh
            self.sigmoid = norm_sigmoid
        else:
            self.tanh = T.tanh
            self.sigmoid = T.nnet.sigmoid

    def allocate_parameters(self, args):
        if hasattr(self, "parameters"):
            return self.parameters

        self.parameters = Empty()

        h0 = theano.shared(zeros((args.num_hidden,)), name="h0")
        c0 = theano.shared(zeros((args.num_hidden,)), name="c0")
        if args.init == "id":
            Wa = theano.shared(np.concatenate([
                np.eye(args.num_hidden),
                orthogonal((args.num_hidden,
                            3 * args.num_hidden)),], axis=1).astype(theano.config.floatX), name="Wa")
        else:
            Wa = theano.shared(orthogonal((args.num_hidden, 4 * args.num_hidden)), name="Wa")
        Wx = theano.shared(orthogonal((nclasses, 4 * args.num_hidden)), name="Wx")
        ab_betas = theano.shared(args.initial_beta  * ones((4 * args.num_hidden,)), name="ab_betas")

        # forget gate bias initialization
        forget_biais = ab_betas.get_value()
        forget_biais[args.num_hidden:2*args.num_hidden] = 1.
        ab_betas.set_value(forget_biais)

        parameters_list = [Wa, Wx, h0, c0, ab_betas]
        if self.norm:
            a_gammas = theano.shared(args.initial_gamma * ones((4 * args.num_hidden,)), name="a_gammas")
            b_gammas = theano.shared(args.initial_gamma * ones((4 * args.num_hidden,)), name="b_gammas")
            c_gammas = theano.shared(args.initial_gamma * ones((args.num_hidden,)), name="c_gammas")
            parameters_list.extend([a_gammas, b_gammas, c_gammas])
        for parameter in parameters_list:
            print parameter.name
            add_role(parameter, PARAMETER)
            setattr(self.parameters, parameter.name, parameter)
        return self.parameters


    def construct_graph(self, args, x, length):
        p = self.allocate_parameters(args)

        if self.norm:
            # Normalize Wa and Wx, and apply gammas
            norm_Wx = p.Wx / T.sqrt((p.Wx**2).sum(axis=0, keepdims=True)) # TODO Check axis
            norm_Wa = p.Wa / T.sqrt((p.Wa**2).sum(axis=0, keepdims=True)) # TODO Check axis
            norm_Wx *= p.b_gammas.dimshuffle('x', 0) # TODO Check axis
            norm_Wa *= p.a_gammas.dimshuffle('x', 0) # TODO Check axis
        else:
            norm_Wx = p.Wx
            norm_Wa = p.Wa
 
        xtilde = T.dot(x, norm_Wx)

        dummy_states = dict(h=T.zeros((xtilde.shape[0], xtilde.shape[1], args.num_hidden)),
                            c=T.zeros((xtilde.shape[0], xtilde.shape[1], args.num_hidden)))
 
        print 'Norm Tanh Gamma'       
        def stepfn(x, dummy_h, dummy_c, h, c, norm_Wa, ab_betas, c_gammas):
            atilde = T.dot(h, norm_Wa)
            btilde = x
            a_normal = atilde
            b_normal = btilde
            if self.norm:
                ab = ab_betas + (a_normal + b_normal) / T.sqrt(2)
            else:
                ab = a_normal + b_normal + ab_betas
            g, f, i, o = [ab[:, j * args.num_hidden:(j + 1) * args.num_hidden]
                          for j in range(4)]
            g = norm_tanh_gamma(g, args.initial_gamma)
            f = T.nnet.sigmoid(f)
            i = T.nnet.sigmoid(i)
            o = T.nnet.sigmoid(o)
            if self.norm:
                c = dummy_c + (f * c + i * g) / T.sqrt(2)
            else:
                c = dummy_c + f * c + i * g
            c_normal = c
            if self.norm:
                h = dummy_h + o * norm_tanh_gamma(c_normal*c_gammas, args.initial_gamma)
            else:
                h = dummy_h + o * T.tanh(c_normal)
            return h, c, atilde, btilde, c_normal


        [h, c, atilde, btilde, htilde], _ = theano.scan(
            stepfn,
            sequences=[xtilde, dummy_states["h"], dummy_states["c"]],
            non_sequences=[norm_Wa, p.ab_betas, p.c_gammas],
            outputs_info=[T.repeat(p.h0[None, :], xtilde.shape[1], axis=0),
                          T.repeat(p.c0[None, :], xtilde.shape[1], axis=0),
                          None, None, None])
        return dict(h=h, c=c, atilde=atilde, btilde=btilde, htilde=htilde), dummy_states


def construct_common_graph(situation, args, outputs, dummy_states, Wy, by, y, y_gammas=None):
    if args.norm:
        ytilde = T.dot(outputs["h"], Wy) / T.sqrt((Wy**2).sum(axis=0, keepdims=True))
        ytilde *= y_gammas.dimshuffle('x', 0)
        ytilde += by
    else:
        ytilde = T.dot(outputs["h"], Wy) + by
    yhat = softmax_lastaxis(ytilde)

    errors = T.neq(T.argmax(y, axis=y.ndim - 1),
                   T.argmax(yhat, axis=yhat.ndim - 1))
    cross_entropies = crossentropy_lastaxes(yhat, y)

    error_rate = errors.mean().copy(name="error_rate")
    cross_entropy = cross_entropies.mean().copy(name="cross_entropy")
    cost = cross_entropy.copy(name="cost")
    graph = ComputationGraph([cost, cross_entropy, error_rate])

    state_grads = dict((k, T.grad(cost, v))
                       for k, v in dummy_states.items())
    extensions = []
    if args.dump:
        if not os.path.exists('/Tmp/laurent/dump'):
            os.makedirs('/Tmp/laurent/dump')
        extensions.append(
            DumpVariables("/Tmp/laurent/dump/%s_hiddens" % situation, graph.inputs,
                          [v.copy(name="%s%s" % (k, suffix))
                           for suffix, things in [("", outputs), ("_grad", state_grads)]
                           for k, v in things.items()],
                          batch=next(get_stream(which_set="train",
                                                batch_size=args.batch_size,
                                                num_examples=args.batch_size,
                                                length=args.length)
                                     .get_epoch_iterator(as_dict=True)),
                          before_training=True, every_n_epochs=5))

    return graph, extensions

def construct_graphs(args, nclasses):
    if args.lstm:
        constructor = LSTM
    else:
        constructor = RNN

    Wy = theano.shared(orthogonal((args.num_hidden, nclasses)), name="Wy")
    by = theano.shared(np.zeros((nclasses,), dtype=theano.config.floatX), name="by")
    for parameter in [Wy, by]:
        add_role(parameter, PARAMETER)
    if args.norm:
        y_gammas = theano.shared(1./1.21 * ones((nclasses,)), name='y_gammas')
        add_role(y_gammas, PARAMETER)
    else:
        y_gammas = None

    x = T.tensor3("features")

    theano.config.compute_test_value = "warn"
    x.tag.test_value = np.random.random((7, args.length, nclasses)).astype(theano.config.floatX)

    # move time axis forward
    x = x.dimshuffle(1, 0, 2)
    # task is to predict next character
    x, y = x[:-1], x[1:]
    length = args.length - 1

    turd = constructor(args, nclasses, args.norm)
    (outputs, dummy_states) = turd.construct_graph(args, x, length)
    graph, extensions = construct_common_graph("training", args, outputs, dummy_states, Wy, by, y, y_gammas)
    args.use_population_statistics = False
    return graph, extensions


if __name__ == "__main__":
    nclasses = 50

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--num-hidden", type=int, default=1000)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--init", choices="id ortho".split(), default="id")
    parser.add_argument("--initial-gamma", type=float, default=1e-1)
    parser.add_argument("--initial-beta", type=float, default=0)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--activation", choices=list(activations.keys()), default="tanh")
    parser.add_argument("--optimizer", choices="sgdmomentum rmsprop adam".split(), default="rmsprop")
    parser.add_argument("--learning-rate-decay", type=float, default=0.0)
    parser.add_argument("--lstm", action="store_true")
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--continue-from")
    args = parser.parse_args()

    if args.path is not None and os.path.exists(args.path):
       print 'Experiment already exists!'
       sys.exit() 

    np.random.seed(args.seed)
    blocks.config.config.default_seed = args.seed

    if args.continue_from:
        from blocks.serialization import load
        main_loop = load(args.continue_from)
        main_loop.run()
        sys.exit(0)

    graph, extensions = construct_graphs(args, nclasses)

    ### optimization algorithm definition
    if args.optimizer == "adam":
        optimizer = Adam(learning_rate=args.learning_rate)
        # zzzz
        #optimizer.learning_rate = theano.shared(np.asarray(optimizer.learning_rate, dtype=theano.config.floatX))
    elif args.optimizer == "rmsprop":
        optimizer = RMSProp(learning_rate=args.learning_rate, decay_rate=0.9)
    elif args.optimizer == "sgdmomentum":
        optimizer = Momentum(learning_rate=args.learning_rate, momentum=0.99)
    step_rule = CompositeRule([
        StepClipping(1.),
        optimizer,
    ])
    algorithm = GradientDescent(cost=graph.outputs[0],
                                parameters=graph.parameters,
                                step_rule=step_rule)
    model = Model(graph.outputs[0])
    
    if args.norm:
        Wa = VariableFilter(theano_name='Wa')(graph.parameters)[0]
        Wx = VariableFilter(theano_name='Wx')(graph.parameters)[0]
        Wy = VariableFilter(theano_name='Wy')(graph.parameters)[0]
        print 'NO FORCE L2 NORM'
        #extensions.append(ForceL2Norm([Wa, Wx, Wy]))

    print 'NO DECAY'
    #extensions.append(SharedVariableModifier(
    #    optimizer.learning_rate,
    #    functools.partial(learning_rate_decayer, args.learning_rate_decay)))

    # step monitor
    step_channels = []
    step_channels.extend([
        algorithm.steps[param].norm(2).copy(name="step_norm:%s" % name)
        for name, param in model.get_parameter_dict().items()])
    step_channels.append(algorithm.total_step_norm.copy(name="total_step_norm"))
    step_channels.append(algorithm.total_gradient_norm.copy(name="total_gradient_norm"))
    step_channels.extend(graph.outputs)
    logger.warning("constructing training data monitor")
    extensions.append(TrainingDataMonitoring(
        step_channels, prefix="iteration", after_batch=False))

    # parameter monitor
    extensions.append(DataStreamMonitoring(
        ([param.norm(2).copy(name="parameter.norm:%s" % name)
          for name, param in model.get_parameter_dict().items()]
         + [optimizer.learning_rate.copy(name="learning_rate")]),
        data_stream=None, after_epoch=True))

    # performance monitor
    for which_set in "train valid test".split():
        logger.warning("constructing %s monitor" % (which_set,))
        channels = list(graph.outputs)
        extensions.append(DataStreamMonitoring(
            channels,
            prefix="%s" % (which_set,), after_epoch=True,
            data_stream=get_stream(which_set=which_set, batch_size=args.batch_size,
                                   num_examples=50000, length=args.length)))

    extensions.extend([
        TrackTheBest("valid_error_rate", "best_valid_error_rate"),
        #DumpBest("best_valid_error_rate", "best.zip"),
        FinishAfter(after_n_epochs=args.num_epochs),
        #FinishIfNoImprovementAfter("best_valid_error_rate", epochs=50),
        #Checkpoint("checkpoint.zip", on_interrupt=False, every_n_epochs=1, use_cpickle=True),
        #DumpLog("log.pkl", after_epoch=True)
        ])

    if not args.cluster:
        extensions.append(ProgressBar())

    extensions.extend([
        Timing(),
        Printing(),
    ])
    if args.path is not None:
        extensions.append(PrintingTo(args.path))
    main_loop = MainLoop(
        data_stream=get_stream(which_set="train", batch_size=args.batch_size, length=args.length, augment=True),
        algorithm=algorithm, extensions=extensions, model=model)
    main_loop.run()
