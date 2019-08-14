from .mlp import *
from .mpnn import *


def get_net(args):
    name = args.network
    if name == 'mlp':
        return Net1
    elif name == 'mpnn':
        return MessagePassingNetwork
