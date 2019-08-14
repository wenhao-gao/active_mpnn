"""
Utility functions for data process
"""
from argparse import Namespace
import csv
from logging import Logger
from typing import List, Set, Tuple

from rdkit import Chem
import numpy as np
from tqdm import tqdm

from dataset.data import MoleculeDatapoint, MoleculeDataset
from feature.utils import load_features


def get_task_names(path: str, use_compound_names: bool = False) -> List[str]:
    """
    Gets the task names from a data CSV file.
    :param path: Path to a CSV file.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :return: A list of task names.
    """
    index = 2 if use_compound_names else 1
    task_names = get_header(path)[index:]

    return task_names


def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.
    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def get_num_tasks(path: str) -> int:
    """
    Gets the number of tasks in a data CSV file.
    :param path: Path to a CSV file.
    :return: The number of tasks.
    """
    return len(get_header(path)) - 1


def get_smiles(path: str, header: bool = True) -> List[str]:
    """
    Returns the smiles strings from a data CSV file (assuming the first line is a header).
    :param path: Path to a CSV file.
    :param header: Whether the CSV file contains a header (that will be skipped).
    :return: A list of smiles strings.
    """
    with open(path) as f:
        reader = csv.reader(f)
        if header:
            next(reader)  # Skip header
        smiles = [line[0] for line in reader]

    return smiles


def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    """
    Filters out invalid SMILES.
    :param data: A MoleculeDataset.
    :return: A MoleculeDataset with only valid molecules.
    """
    return MoleculeDataset([datapoint for datapoint in data
                            if datapoint.smiles != '' and datapoint.mol is not None
                            and datapoint.mol.GetNumHeavyAtoms() > 0])


def get_data(path: str = None,
             skip_invalid_smiles: bool = True,
             args: Namespace = None,
             features_path: List[str] = None,
             max_data_size: int = None,
             use_compound_names: bool = None,
             logger: Logger = None) -> MoleculeDataset:
    """
    Gets smiles string and target values (and optionally compound names if provided) from a CSV file.
    :param path: Path to a CSV file.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :param args: Arguments.
    :param features_path: A list of paths to files containing features. If provided, it is used
    in place of args.features_path.
    :param max_data_size: The maximum number of data points to load.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :param logger: Logger.
    :return: A MoleculeDataset containing smiles strings and target values along
    with other info such as additional features and compound names when desired.
    """
    debug = logger.debug if logger is not None else print

    if path is None:
        path = args.init_data
        path_pool = args.pool_data
    else:
        path_pool = None

    if args is not None:
        # Prefer explicit function arguments but default to args if not provided
        features_path = features_path if features_path is not None else args.features_path
        max_data_size = max_data_size if max_data_size is not None else args.max_data_size
        use_compound_names = use_compound_names if use_compound_names is not None else args.use_compound_names
    else:
        use_compound_names = False

    max_data_size = max_data_size or float('inf')

    # Load features
    if features_path is not None:
        features_data = []
        for feat_path in features_path:
            features_data.append(load_features(feat_path))  # each is num_data x num_features
        features_data = np.concatenate(features_data, axis=1)
    else:
        features_data = None

    skip_smiles = set()

    # Load data
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        target_idx = header.index(args.mol_prop)
        idxs = [0, target_idx]

        lines = []
        for line in reader:
            smiles = line[0]

            if smiles in skip_smiles:
                continue

            lines.append(list(np.array(line)[idxs]))

            if len(lines) >= max_data_size:
                break

    if path_pool is not None:
        with open(path_pool) as f:
            reader = csv.reader(f)
            next(reader)

            for line in reader:
                smiles = line[0]

                if smiles in skip_smiles:
                    continue

                lines.append([smiles, 0])

                if len(lines) >= max_data_size:
                    break

    data = MoleculeDataset([
        MoleculeDatapoint(
            line=line,
            args=args,
            features=features_data[i] if features_data is not None else None,
            use_compound_names=use_compound_names
        ) for i, line in tqdm(enumerate(lines), total=len(lines))
    ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    if data.data[0].features is not None:
        args.features_dim = len(data.data[0].features)

    return data


def get_data_from_smiles(smiles: List[str], skip_invalid_smiles: bool = True, logger: Logger = None) -> MoleculeDataset:
    """
    Converts SMILES to a MoleculeDataset.
    :param smiles: A list of SMILES strings.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :param logger: Logger.
    :return: A MoleculeDataset with all of the provided SMILES.
    """
    debug = logger.debug if logger is not None else print

    data = MoleculeDataset([MoleculeDatapoint([smile]) for smile in smiles])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data


def get_class_sizes(data: MoleculeDataset) -> List[List[float]]:
    """
    Determines the proportions of the different classes in the classification dataset.
    :param data: A classification dataset
    :return: A list of lists of class proportions. Each inner list contains the class proportions
    for a task.
    """
    targets = data.targets()

    # Filter out Nones
    valid_targets = [[] for _ in range(data.num_tasks())]
    for i in range(len(targets)):
        for task_num in range(len(targets[i])):
            if targets[i][task_num] is not None:
                valid_targets[task_num].append(targets[i][task_num])

    class_sizes = []
    for task_targets in valid_targets:
        # Make sure we're dealing with a binary classification task
        assert set(np.unique(task_targets)) <= {0, 1}

        try:
            ones = np.count_nonzero(task_targets) / len(task_targets)
        except ZeroDivisionError:
            ones = float('nan')
            print('Warning: class has no targets')
        class_sizes.append([1 - ones, ones])

    return class_sizes


def validate_data(data_path: str) -> Set[str]:
    """
    Validates a data CSV file, returning a set of errors.
    :param data_path: Path to a data CSV file.
    :return: A set of error messages.
    """
    errors = set()

    header = get_header(data_path)

    with open(data_path) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        smiles, targets = [], []
        for line in reader:
            smiles.append(line[0])
            targets.append(line[1:])

    # Validate header
    if len(header) == 0:
        errors.add('Empty header')
    elif len(header) < 2:
        errors.add('Header must include task names.')

    mol = Chem.MolFromSmiles(header[0])
    if mol is not None:
        errors.add('First row is a SMILES string instead of a header.')

    # Validate smiles
    for smile in tqdm(smiles, total=len(smiles)):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            errors.add('Data includes an invalid SMILES.')

    # Validate targets
    num_tasks_set = set(len(mol_targets) for mol_targets in targets)
    if len(num_tasks_set) != 1:
        errors.add('Inconsistent number of tasks for each molecule.')

    if len(num_tasks_set) == 1:
        num_tasks = num_tasks_set.pop()
        if num_tasks != len(header) - 1:
            errors.add('Number of tasks for each molecule doesn\'t match number of tasks in header.')

    unique_targets = set(np.unique([target for mol_targets in targets for target in mol_targets]))

    if unique_targets <= {''}:
        errors.add('All targets are missing.')

    for target in unique_targets - {''}:
        try:
            float(target)
        except ValueError:
            errors.add('Found a target which is not a number.')

    return errors