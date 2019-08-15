from argparse import Namespace

import pandas as pd
import numpy as np

from dataset.utils import get_data


def get_dataset(args: Namespace = None):
    """
    Read in a molecule data set
    :param args: Arguments
    :return: training data, test data, index_labeled, labeled_number
    """
    assert args is not None

    if isinstance(args.max_data_size, str):
        args.max_data_size = int(args.max_data_size)

    df_init = pd.read_csv(args.init_data)
    df_test = pd.read_csv(args.test_data)
    df_pool = pd.read_csv(args.pool_data)

    n_label = len(df_init)
    n_pool = len(df_pool)
    n_test = len(df_test)

    idxs_lb = np.zeros(n_label + n_pool, dtype=bool)
    idxs_tmp = np.arange(n_label + n_pool)
    idxs_lb[idxs_tmp[:n_label]] = True

    data_train = get_data(args=args, skip_invalid_smiles=False, max_data_size=args.max_data_size)
    data_test = get_data(path=args.test_data, args=args, skip_invalid_smiles=False)

    return data_train, data_test, idxs_lb[:args.max_data_size], n_test
