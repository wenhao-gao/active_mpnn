from argparse import Namespace

import os
import logging
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from dataset.get_data import get_dataset
from model.get_models import get_net
from model.nn_utils import initialize_weights
from oracle.get_oracles import get_oracle
from query_strategies.get_strategy import get_strategy
from parsing import parse_args


def train_model(
        args: Namespace = None,
        logger: logging.Logger = None,
        writer: SummaryWriter = None):
    """
    Start a active training process

    :param args: arguments
    :param logger: logger
    :param writer: tensorboard writer
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False

    # Construct data set
    data_train, data_test, idxs_lb, n_test = get_dataset(args)
    info('number of labeled pool: {}'.format(sum(idxs_lb)))
    info('number of unlabeled pool: {}'.format(len(data_train) - sum(idxs_lb)))
    info('number of testing pool: {}'.format(n_test))

    net = get_net(args)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    net = net(args, device)
    initialize_weights(net)
    if args.parameters is not None:
        net.load_state_dict(torch.load(args.param))
    # optimizer = optim.SGD(net.parameters(),
    #                       lr=args.learning_rate,
    #                       momentum=0.5
    #                       )
    optimizer = optim.Adam(
            params=net.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta_1, args.adam_beta_2),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False
        )
    lr_schedule = optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=args.learning_rate_decay_rate
    )
    oracle = get_oracle(args)
    strategy = get_strategy(args)
    strategy = strategy(data_train, idxs_lb, net, optimizer, lr_schedule, args, logger, writer)

    info('SEED {}'.format(args.seed))
    info(type(strategy).__name__)
    losses = []
    n_iter = 0

    n_iter = strategy.train(n_iter)
    mse, mae, maxae = strategy.evaluate(data_test)
    losses.append(mae)
    info('Round 0\ntesting MAE {}'.format(mae))

    if writer is not None:
        writer.add_scalar('query_mse', mse, 0)
        writer.add_scalar('query_mae', mae, 0)
        writer.add_scalar('query_maxae', maxae, 0)

    for rd in range(1, args.round + 1):
        info('Round {}'.format(rd))

        # query
        q_idxs = strategy.query(args.query)
        data_train, idxs_lb = oracle(data_train, idxs_lb, q_idxs)

        # update
        strategy.update(idxs_lb)
        n_iter = strategy.train(n_iter)

        # round accuracy
        mse, mae, maxae = strategy.evaluate(data_test)
        losses.append(mae)
        info('Round {}\ntesting MAE {}'.format(rd, mae))
        if writer is not None:
            writer.add_scalar('query_mse', mse, rd)
            writer.add_scalar('query_mae', mae, rd)
            writer.add_scalar('query_maxae', maxae, rd)
        strategy.save_net(rd)

    # print results
    info('SEED {}'.format(args.seed))
    info(type(strategy).__name__)
    info(losses)


def set_logger(logger: logging.Logger, save_dir: str = None, quiet: bool = False):
    """
    Sets up a logger with a stream handler and two file handlers.
    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.
    :param logger: A logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e. print only important info).
    """
    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)


if __name__ == '__main__':
    args = parse_args()

    logger = logging.getLogger('train')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    set_logger(logger, args.log_path, args.quiet)

    try:
        writer = SummaryWriter(log_dir=args.log_path)
    except:
        writer = SummaryWriter(logdir=args.log_path)

    train_model(args, logger, writer)
