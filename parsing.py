"""
Code handle the arguments
"""
from argparse import ArgumentParser, Namespace
import json
import os


def add_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('-p', '--parameters', default=None,
                        help='The network parameters to begin with.')
    parser.add_argument('-t', '--task', default='test',
                        help='The task name.')
    parser.add_argument('-c', '--path_to_config', default=None,
                        help='The JSON file define the hyper parameters.')
    parser.add_argument('-m', '--log_path', default='./checkpoints/',
                        help='path to put output files.')
    parser.add_argument('-v', '--quiet', action='store_true', default=True,
                        help='Whether to see the intermediate information.')
    parser.add_argument('--save_frequency', default=1,
                        help='The frequency to save a checkpoint.')

    # data specify
    parser.add_argument('--init_data', default='data/sheridan_train.csv',
                        help='The path to initially labeled data file.')
    parser.add_argument('--pool_data', default='data/chembl250k.csv',
                        help='The path to candidate pool.')
    parser.add_argument('--test_data', default='data/sheridan_test.csv',
                        help='The path to test data file.')
    parser.add_argument('--mol_col', default='SMILES',
                        help='The column contains the molecules.')
    parser.add_argument('--mol_prop', default='sa_score',
                        help='The column contains the property to predict.')

    # active learning strategy
    parser.add_argument('--seed', default=123,
                        help='The random seed to be used.')
    parser.add_argument('--round', default=20,
                        help='The number of rounds to make a query.')
    parser.add_argument('--query', default=128,
                        help='The number of samples to query every time.')
    parser.add_argument('--strategy', default='maxmin',
                        choices=['random', 'maxmin', 'kcenter', 'variance', 'kmean'],
                        help='Name a query strategy')
    parser.add_argument('--oracle', default='sa',
                        choices=['sa', 'sc', 'tb', 'smiles', 'test'],
                        help='Name a oracle to learn from')
    parser.add_argument('--max_data_size', default=999999999,
                        help='The number of samples to query every time.')

    # Network arguments
    parser.add_argument('--network', default='mpnn',
                        choices=['mpnn', 'mlp', 'regression'],
                        help='The q funciton.')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')
    parser.add_argument('--optimizer', default='Adam',
                        help='The opitmizer to use.')
    parser.add_argument('--batch_norm', action='store_true', default=True,
                        help='Use the batch normalization or not.')
    parser.add_argument('--dropout', default=0.5,
                        help='The dropout probability.')
    parser.add_argument('--n_drop', default=25,
                        help='The dropout probability.')
    parser.add_argument('--message_dropout', default=0,
                        help='The dropout probability.')
    parser.add_argument('--adam_beta_1', default=0.9,
                        help='The beta_1 in adam optimizer.')
    parser.add_argument('--adam_beta_2', default=0.999,
                        help='The beta_2 in adam optimizer.')
    parser.add_argument('--grad_clipping', default=10,
                        help='The gradient clipping.')
    parser.add_argument('--learning_rate', default=1e-4,
                        help='The learning rate to begin with.')
    parser.add_argument('--learning_rate_decay_steps', default=10000,
                        help='Learning rate decay steps.')
    parser.add_argument('--learning_rate_decay_rate', default=0.8,
                        help='Learning rate decay rate.')
    parser.add_argument('--epoch', default=2,
                        help='The number of epoches to train.')
    parser.add_argument('--transform', default=None,
                        help='The number of samples to query every time.')
    parser.add_argument('--loader_tr_args', default={'batch_size': 64, 'num_workers': 1},
                        help='The number of samples to query every time.')
    parser.add_argument('--loader_te_args', default={'batch_size': 1000, 'num_workers': 1},
                        help='The number of samples to query every time.')
    parser.add_argument('--features_generator', default=['morgan'],
                        help='Path to feature files')
    parser.add_argument('--features_path', default=None,
                        help='Path to feature files')
    parser.add_argument('--use_compound_names', action='store_true', default=False,
                        help='Whether to see the intermediate information.')
    parser.add_argument('--batch_size', default=128,
                        help='The number of samples to query every time.')
    parser.add_argument('--log_frequency', default=1,
                        help='The number of samples to query every time.')
    parser.add_argument('--fingerprint_radius', default=3,
                        help='The Morgan fingerprint radius.')
    parser.add_argument('--fingerprint_length', default=2048,
                        help='The Morgan fingerprint length.')
    parser.add_argument('--dense_layers', default=[1024, 512, 128, 32],
                        help='The dense layers of ffn.')
    parser.add_argument('--hidden_size', default=300,
                        help='The hidden vector size.')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='To use bias in graph network or not.')
    parser.add_argument('--depth', default=3,
                        help='The message passing depth of graph network.')
    parser.add_argument('--ffn_hidden_size', default=300,
                        help='The hidden size of following ffn.')
    parser.add_argument('--ffn_num_layers', default=2,
                        help='The number of layers of following ffn.')
    parser.add_argument('--output_dim', default=1,
                        help='The number of output dimension.')
    parser.add_argument('--atom_messages', action='store_true', default=False,
                        help='To use atom features instead of concatenation of atom and bond')
    parser.add_argument('--feature_only', action='store_true', default=True,
                        help='Only use the artificial features.')
    parser.add_argument('--use_input_features', action='store_true', default=True,
                        help='Concatenate input features.')
    parser.add_argument('--features_dim', default=1,
                        help='The feature dimension.')
    parser.add_argument('--no_cache', action='store_true', default=False,
                        help='Turn off caching mol2graph computation.')


def modify_args(args: Namespace):
    """Modify the arguments and read json configuration file to overwrite."""
    hparams = {}
    if args.path_to_config is not None:
        with open(args.path_to_config, 'r') as f:
            hparams.update(json.load(f))

        for key, value in hparams.items():
            setattr(args, key, value)
    return args


def parse_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    modify_args(args)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    args.log_path = os.path.join(args.log_path, args.task)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    return args


if __name__ == "__main__":
    args = parse_args()
    print(type(args.atom_types))
