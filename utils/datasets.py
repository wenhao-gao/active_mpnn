from __future__ import print_function, division

import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from utils.functions import read_sdf, read_xyz


class Alchemy(Dataset):

    def __init__(self, root, csv_file):
        self.frame = pd.read_csv(csv_file)
        self.root = root

    def __getitem__(self, idx):
        gdb_idx = self.frame.iloc[idx, 0]
        filename = str(gdb_idx) + '.sdf'
        path_to_file = os.path.join(self.root, filename)
        atoms, atom_coords, dist_mat, adja_mat = read_sdf(path_to_file)

        sample = {
            'atoms': atoms,
            'coords': atom_coords,
            'dist_mat': dist_mat,
            'adja_mat': adja_mat,
            'prop_0': self.frame.iloc[idx, 1],
            'prop_1': self.frame.iloc[idx, 2],
            'prop_2': self.frame.iloc[idx, 3],
            'prop_3': self.frame.iloc[idx, 4],
            'prop_4': self.frame.iloc[idx, 5],
            'prop_5': self.frame.iloc[idx, 6],
            'prop_6': self.frame.iloc[idx, 7],
            'prop_7': self.frame.iloc[idx, 8],
            'prop_8': self.frame.iloc[idx, 9],
            'prop_9': self.frame.iloc[idx, 10],
            'prop_10': self.frame.iloc[idx, 11],
            'prop_11': self.frame.iloc[idx, 12]
        }

        return (sample['adja_mat'], sample['atoms'], sample['dist_mat']), sample['prop_0']

    def __len__(self):
        return len(self.frame)

    def set_target_transform(self, target_transform):
        self.target_transform = target_transform


class CSC(Dataset):

    def __init__(self, root, csv_file):
        self.frame = pd.read_csv(csv_file)
        self.root = root

    def __getitem__(self, idx):
        molecule_name = self.frame.iloc[idx, 0]
        filename = str(molecule_name) + '.xyz'
        path_to_file = os.path.join(self.root, filename)
        atoms, atom_coords, dist_mat = read_xyz(path_to_file)

        sample = {
            'atoms': atoms,
            'coords': atom_coords,
            'dist_mat': dist_mat,
            'atom_index_0': self.frame.iloc[idx, 1],
            'atom_index_1': self.frame.iloc[idx, 2],
            'type': self.frame.iloc[idx, 3],
            'scc': self.frame.iloc[idx, 4]
        }

        return sample

    def __len__(self):
        return len(self.frame)

    def set_target_transform(self, target_transform):
        self.target_transform = target_transform


def dist_mat2e(dist_mat, adja_mat):
    e = {}
    na = len(dist_mat)
    pass
