import sys
import logging
from collections import OrderedDict
import numpy as np
import theano, theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import function
import blocks.config
import fuel.datasets, fuel.streams, fuel.transformers, fuel.schemes

### optimization algorithm definition
from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, RMSProp, StepClipping, CompositeRule, Momentum
from blocks.model import Model
from blocks.extensions import FinishAfter, Printing, ProgressBar, Timing, SimpleExtension
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.training import TrackTheBest
from blocks.extensions.saveload import Checkpoint
from extensions import DumpLog, DumpBest, PrintingTo, DumpVariables
from blocks.main_loop import MainLoop
from blocks.utils import shared_floatx_zeros
from blocks.roles import add_role, PARAMETER
from blocks.filter import VariableFilter

import util


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

_datasets = None
def get_dataset(which_set):
    global _datasets
    if not _datasets:
        MNIST = fuel.datasets.MNIST
        # jump through hoops to instantiate only once and only if needed
        _datasets = dict(
            train=MNIST(which_sets=["train"], subset=slice(None, 50000)),
            valid=MNIST(which_sets=["train"], subset=slice(50000, None)),
            test=MNIST(which_sets=["test"]))
    return _datasets[which_set]

def get_stream(which_set, batch_size, num_examples=None):
    dataset = get_dataset(which_set)
    if num_examples is None or num_examples > dataset.num_examples:
        num_examples = dataset.num_examples
    stream = fuel.streams.DataStream.default_stream(
        dataset,
        iteration_scheme=fuel.schemes.ShuffledScheme(num_examples, batch_size))
    return stream


class Empty(object):
    pass


def norm_tanh(x):
    y = T.tanh(x)
    return y / T.sqrt(0.394)


def norm_sigmoid(x):
    y = T.nnet.sigmoid(x)
    return (y - 0.5) / T.sqrt(0.043)


class LSTM(object):
    def __init__(self, args, nclasses, norm=False):
        self.nclasses = nclasses
        self.norm = norm
        if norm:
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
        Wx = theano.shared(orthogonal((1, 4 * args.num_hidden)), name="Wx")
        ab_betas = theano.shared(args.initial_beta  * ones((4 * args.num_hidden,)), name="ab_betas")

        # forget gate bias initialization
        forget_biais = ab_betas.get_value()
        forget_biais[args.num_hidden:2*args.num_hidden] = 1.
        ab_betas.set_value(forget_biais)

        parameters_list = [Wa, Wx, h0, c0, ab_betas]
        if self.norm:
            a_gammas = theano.shared(args.initial_gamma * ones((4 * args.num_hidden,)), name="a_gammas")
            b_gammas = theano.shared(args.initial_gamma * ones((4 * args.num_hidden,)), name="b_gammas")
            parameters_list.extend([a_gammas, b_gammas])
        for parameter in parameters_list:
            print parameter.name
            add_role(parameter, PARAMETER)
            setattr(self.parameters, parameter.name, parameter)
        return self.parameters


    def construct_graph_ref(self, args, x, length):
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

        if args.noise:
            # prime h with white noise
            Trng = MRG_RandomStreams()
            h_prime = Trng.normal((xtilde.shape[1], args.num_hidden), std=args.noise)
        elif args.summarize:
            # prime h with mean of example
            h_prime = x.mean(axis=[0, 2])[:, None]
        else:
            h_prime = 0

        dummy_states = dict(h=T.zeros((xtilde.shape[0], xtilde.shape[1], args.num_hidden)),
                            c=T.zeros((xtilde.shape[0], xtilde.shape[1], args.num_hidden)))
        
        def stepfn(x, dummy_h, dummy_c, h, c, norm_Wa):
            atilde = T.dot(h, norm_Wa)
            btilde = x
            a_normal = atilde
            b_normal = btilde
            if self.norm:
                ab = (a_normal + b_normal) / T.sqrt(2) + p.ab_betas
            else:
                ab = a_normal + b_normal + p.ab_betas
            g, f, i, o = [fn(ab[:, j * args.num_hidden:(j + 1) * args.num_hidden])
                          for j, fn in enumerate([self.tanh] + 3 * [self.sigmoid])]
            if self.norm:
                c = dummy_c + (f * c + i * g) / T.sqrt(2)
            else:
                c = dummy_c + f * c + i * g
            c_normal = c
            h = dummy_h + o * self.tanh(c_normal)
            return h, c, atilde, btilde, c_normal


        [h, c, atilde, btilde, htilde], _ = theano.scan(
            stepfn,
            sequences=[xtilde, dummy_states["h"], dummy_states["c"]],
            non_sequences=[norm_Wa],
            outputs_info=[T.repeat(p.h0[None, :], xtilde.shape[1], axis=0) + h_prime,
                          T.repeat(p.c0[None, :], xtilde.shape[1], axis=0),
                          None, None, None])
        return dict(h=h, c=c, atilde=atilde, btilde=btilde, htilde=htilde), dummy_states


def construct_common_graph(situation, args, outputs, dummy_states, Wy, by, y):
    ytilde = T.dot(outputs["h"][-1], Wy) + by
    yhat = T.nnet.softmax(ytilde)

    errors = T.neq(y, T.argmax(yhat, axis=1))
    cross_entropies = T.nnet.categorical_crossentropy(yhat, y)

    error_rate = errors.mean().copy(name="error_rate")
    cross_entropy = cross_entropies.mean().copy(name="cross_entropy")
    cost = cross_entropy.copy(name="cost")
    graph = ComputationGraph([cost, cross_entropy, error_rate])

    state_grads = dict((k, T.grad(cost, v)) for k, v in dummy_states.items())

    extensions = []
    # extensions = [
    #     DumpVariables("%s_hiddens" % situation, graph.inputs,
    #                   [v.copy(name="%s%s" % (k, suffix))
    #                    for suffix, things in [("", outputs), ("_grad", state_grads)]
    #                    for k, v in things.items()],
    #                   batch=next(get_stream(which_set="train",
    #                                         batch_size=args.batch_size,
    #                                         num_examples=args.batch_size)
    #                              .get_epoch_iterator(as_dict=True)),
    #                   before_training=True, every_n_epochs=10)]

    return graph, extensions

def construct_graphs(args, nclasses, length):
    constructor = LSTM

    if args.permuted:
        permutation = np.random.randint(0, length, size=(length,))

    Wy = theano.shared(orthogonal((args.num_hidden, nclasses)), name="Wy")
    by = theano.shared(np.zeros((nclasses,), dtype=theano.config.floatX), name="by")

    ### graph construction
    inputs = dict(features=T.tensor4("features"), targets=T.imatrix("targets"))
    x, y = inputs["features"], inputs["targets"]

    theano.config.compute_test_value = "warn"
    batch = next(get_stream(which_set="train", batch_size=args.batch_size).get_epoch_iterator())
    x.tag.test_value = batch[0]
    y.tag.test_value = batch[1]

    x = x.reshape((x.shape[0], length + 0, 1))
    y = y.flatten(ndim=1)
    x = x.dimshuffle(1, 0, 2)
    x = x[0:, :, :]

    if args.permuted:
        x = x[permutation]

    turd = constructor(args, nclasses, args.norm)
    (outputs, dummy_states) = turd.construct_graph_ref(args, x, length)
    graph, extensions = construct_common_graph("training", args, outputs, dummy_states, Wy, by, y)
    add_role(Wy, PARAMETER)
    add_role(by, PARAMETER)
    args.use_population_statistics = False
    return graph, extensions 


if __name__ == "__main__":
    sequence_length = 784
    nclasses = 10

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--num-hidden", type=int, default=100)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--lstm", action="store_true")
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--initial-gamma", type=float, default=0.1)
    parser.add_argument("--initial-beta", type=float, default=0)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--init", type=str, default="ortho")
    parser.add_argument("--continue-from")
    parser.add_argument("--permuted", action="store_true")
    args = parser.parse_args()

    #assert not (args.noise and args.summarize)
    np.random.seed(args.seed)
    blocks.config.config.default_seed = args.seed


    if args.continue_from:
        from blocks.serialization import load
        main_loop = load(args.continue_from)
        main_loop.run()
        sys.exit(0)

    graph, extensions = construct_graphs(args, nclasses, sequence_length)

    ### optimization algorithm definition
    step_rule = CompositeRule([
        StepClipping(1.),
        #Momentum(learning_rate=args.learning_rate, momentum=0.9),
        RMSProp(learning_rate=args.learning_rate, decay_rate=0.5),
    ])

    algorithm = GradientDescent(cost=graph.outputs[0],
                                parameters=graph.parameters,
                                step_rule=step_rule)
    model = Model(graph.outputs[0])
    
    extensions = []
    if args.norm:
        Wa = VariableFilter(theano_name='Wa')(graph.parameters)[0]
        Wx = VariableFilter(theano_name='Wx')(graph.parameters)[0]
        extensions.append(ForceL2Norm([Wa, Wx]))

    # step monitor (after epoch to limit the log size)
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
        [param.norm(2).copy(name="parameter.norm:%s" % name)
         for name, param in model.get_parameter_dict().items()],
        data_stream=None, after_epoch=True))

    # performance monitor
    for which_set in "train valid test".split():
        logger.warning("constructing %s monitor" % (which_set,))
        channels = list(graph.outputs)
        extensions.append(DataStreamMonitoring(
            channels,
            prefix="%s" % (which_set,), after_epoch=True,
            data_stream=get_stream(which_set=which_set, batch_size=args.batch_size)))#, num_examples=1000)))

    extensions.extend([
        TrackTheBest("valid_error_rate", "best_valid_error_rate"),
        DumpBest("best_valid_error_rate", "best.zip"),
        FinishAfter(after_n_epochs=args.num_epochs),
        #FinishIfNoImprovementAfter("best_valid_error_rate", epochs=50),
        Checkpoint("checkpoint.zip", on_interrupt=False, every_n_epochs=1, use_cpickle=True),
        DumpLog("log.pkl", after_epoch=True)])

    if not args.cluster:
        extensions.append(ProgressBar())


    extensions.extend([
        Timing(),
        Printing(),
        PrintingTo("log"),
    ])
    main_loop = MainLoop(
        data_stream=get_stream(which_set="train", batch_size=args.batch_size),
        algorithm=algorithm, extensions=extensions, model=model)
    main_loop.run()
