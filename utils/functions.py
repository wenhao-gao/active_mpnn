"""
The implementation for AlChemy contest
QM properties prediction
-----------------------------------------
Message Passing Neural Network Implementation

Utilities
"""
import rdkit
import torch
from joblib import Parallel, delayed
import multiprocessing
import networkx as nx
import numpy as np
import shutil
import os


def calc_dist_mat(na, coords):
    dist_mat = np.zeros((na, na))
    for i in range(1, na):
        for j in range(i):
            coord_1 = coords[i]
            coord_2 = coords[j]
            dist_mat[i][j] = np.sqrt(sum(np.square(coord_1 - coord_2)))
    return dist_mat + dist_mat.T


def read_sdf(path_to_file):
    atom_one_hot = {
        'H': np.array([1, 0, 0, 0, 0, 0, 0]),
        'C': np.array([0, 1, 0, 0, 0, 0, 0]),
        'N': np.array([0, 0, 1, 0, 0, 0, 0]),
        'O': np.array([0, 0, 0, 1, 0, 0, 0]),
        'F': np.array([0, 0, 0, 0, 1, 0, 0]),
        'S': np.array([0, 0, 0, 0, 0, 1, 0]),
        'Cl': np.array([0, 0, 0, 0, 0, 0, 1])
    }
    with open(path_to_file, 'r') as f:
        for i in range(4):
            line = f.readline()

        num_atom = int(line.split()[0])

        atoms = []
        atom_coords = []

        for i in range(num_atom):
            line = f.readline().split()
            atom_coords.append(np.array([float(x) for x in line[:3]]))
            atoms.append(atom_one_hot[line[3]])

        adja_mat = np.zeros((num_atom, num_atom))
        dist_mat = calc_dist_mat(num_atom, atom_coords)

        while True:
            line = f.readline().split()
            if line[0] == 'M':
                break
            adja_mat[int(line[0]) - 1][int(line[1]) - 1] = int(line[2])

    return np.array(atoms), np.array(atom_coords), dist_mat, adja_mat + adja_mat.T


def read_xyz(path_to_file):
    atom_one_hot = {
        'H': np.array([1, 0, 0, 0, 0]),
        'C': np.array([0, 1, 0, 0, 0]),
        'N': np.array([0, 0, 1, 0, 0]),
        'O': np.array([0, 0, 0, 1, 0]),
        'F': np.array([0, 0, 0, 0, 1])
    }

    with open(path_to_file, 'r') as f:
        num_atom = int(f.readline().split()[0])
        _ = f.readline()
        atoms = []
        atom_coords = []

        for i in range(num_atom):
            line = f.readline().split()
            atoms.append(atom_one_hot[line[0]])
            atom_coords.append([float(x) for x in line[1:]])

        atoms = np.array(atoms)
        atom_coords = np.array(atom_coords)
        return atoms, atom_coords, calc_dist_mat(num_atom, atom_coords)


def qm9_nodes(g, hydrogen=False):
    h = []
    for n, d in g.nodes_iter(data=True):
        h_t = []
        # Atom type (One-hot H, C, N, O F)
        h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F']]
        # Atomic number
        h_t.append(d['a_num'])
        # Partial Charge
        h_t.append(d['pc'])
        # Acceptor
        h_t.append(d['acceptor'])
        # Donor
        h_t.append(d['donor'])
        # Aromatic
        h_t.append(int(d['aromatic']))
        # Hybradization
        h_t += [int(d['hybridization'] == x) for x in
                [rdkit.Chem.rdchem.HybridizationType.SP, rdkit.Chem.rdchem.HybridizationType.SP2,
                 rdkit.Chem.rdchem.HybridizationType.SP3]]
        # If number hydrogen is used as a
        if hydrogen:
            h_t.append(d['num_h'])
        h.append(h_t)
    return h


def qm9_edges(g, e_representation='raw_distance'):
    remove_edges = []
    e = {}
    for n1, n2, d in g.edges_iter(data=True):
        e_t = []
        # Raw distance function
        if e_representation == 'chem_graph':
            if d['b_type'] is None:
                remove_edges += [(n1, n2)]
            else:
                e_t += [i + 1 for i, x in
                        enumerate([rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE,
                                   rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.AROMATIC])
                        if x == d['b_type']]
        elif e_representation == 'distance_bin':
            if d['b_type'] is None:
                step = (6 - 2) / 8.0
                start = 2
                b = 9
                for i in range(0, 9):
                    if d['distance'] < (start + i * step):
                        b = i
                        break
                e_t.append(b + 5)
            else:
                e_t += [i + 1 for i, x in
                        enumerate([rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE,
                                   rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.AROMATIC])
                        if x == d['b_type']]
        elif e_representation == 'raw_distance':
            if d['b_type'] is None:
                remove_edges += [(n1, n2)]
            else:
                e_t.append(d['distance'])
                e_t += [int(d['b_type'] == x) for x in
                        [rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE,
                         rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.AROMATIC]]
        else:
            print('Incorrect Edge representation transform')
            quit()
        if e_t:
            e[(n1, n2)] = e_t
    for edg in remove_edges:
        g.remove_edge(*edg)
    return nx.to_numpy_matrix(g), e


def normalize_data(data, mean, std):
    data_norm = (data - mean) / std
    return data_norm


def get_values(obj, start, end, prop):
    vals = []
    for i in range(start, end):
        v = {}
        if 'degrees' in prop:
            v['degrees'] = set(sum(obj[i][0][0].sum(axis=0, dtype='int').tolist(), []))
        if 'edge_labels' in prop:
            v['edge_labels'] = set(sum(list(obj[i][0][2].values()), []))
        if 'target_mean' in prop or 'target_std' in prop:
            v['params'] = obj[i][1]
        vals.append(v)
    return vals


def get_graph_stats(graph_obj_handle, prop='degrees'):
    # if prop == 'degrees':
    num_cores = multiprocessing.cpu_count()
    inputs = [int(i * len(graph_obj_handle) / num_cores) for i in range(num_cores)] + [len(graph_obj_handle)]
    res = Parallel(n_jobs=num_cores)(
        delayed(get_values)(graph_obj_handle, inputs[i], inputs[i + 1], prop) for i in range(num_cores))

    stat_dict = {}

    if 'degrees' in prop:
        stat_dict['degrees'] = list(set([d for core_res in res for file_res in core_res for d in file_res['degrees']]))
    if 'edge_labels' in prop:
        stat_dict['edge_labels'] = list(
            set([d for core_res in res for file_res in core_res for d in file_res['edge_labels']]))
    if 'target_mean' in prop or 'target_std' in prop:
        param = np.array([file_res['params'] for core_res in res for file_res in core_res])
    if 'target_mean' in prop:
        stat_dict['target_mean'] = np.mean(param, axis=0)
    if 'target_std' in prop:
        stat_dict['target_std'] = np.std(param, axis=0)

    return stat_dict


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred = pred.type_as(target)
    target = target.type_as(pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def collate_g(batch):
    batch_sizes = np.max(np.array([[len(input_b[1]), len(input_b[1][0]), len(input_b[2]),
                                    len(list(input_b[2].values())[0])]
                                   if input_b[2] else
                                   [len(input_b[1]), len(input_b[1][0]), 0, 0]
                                   for (input_b, target_b) in batch]), axis=0)

    g = np.zeros((len(batch), batch_sizes[0], batch_sizes[0]))
    h = np.zeros((len(batch), batch_sizes[0], batch_sizes[1]))
    e = np.zeros((len(batch), batch_sizes[0], batch_sizes[0], batch_sizes[3]))

    target = np.zeros((len(batch), len(batch[0][1])))

    for i in range(len(batch)):

        num_nodes = len(batch[i][0][1])

        # Adjacency matrix
        g[i, 0:num_nodes, 0:num_nodes] = batch[i][0][0]

        # Node features
        h[i, 0:num_nodes, :] = batch[i][0][1]

        # Edges
        for edge in batch[i][0][2].keys():
            e[i, edge[0], edge[1], :] = batch[i][0][2][edge]
            e[i, edge[1], edge[0], :] = batch[i][0][2][edge]

        # Target
        target[i, :] = batch[i][1]

    g = torch.FloatTensor(g)
    h = torch.FloatTensor(h)
    e = torch.FloatTensor(e)
    target = torch.FloatTensor(target)

    return g, h, e, target


def save_checkpoint(state, is_best, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_model_file)


