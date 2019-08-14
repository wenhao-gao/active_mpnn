import numpy as np
import requests
import json
import pickle
from oracle import sascorer
import rdkit.Chem as Chem
# from scscore.standalone_model_numpy import SCScorer


def get_oracle(args):
    name = args.oracle
    if name == 'tb':
        return tb_score
    elif name == 'sa':
        return label_sa_score
    # elif name == 'sc':
    #     return sc_score
    # elif name == 'test':
    #     return test_label


def label_sa_score(data, idxs_lb, q_idxs):
    """
    Label the data with query index
    :param data: training data pool
    :param idxs_lb: labeled index
    :param q_idxs: query index
    :return: labeled dataset, new labeled index
    """
    idxs_lb[q_idxs] = True
    for i in range(len(q_idxs)):
        data[q_idxs][i].targets = gaussian_wrapper(sa_score(data[q_idxs][i].smiles))
    return data, idxs_lb


def tb_score(smi):
    pass


def get_buyability(molecule):
    """
    Check if the molecule is buyable
    :param molecule: inquiry molecule in format of SMILES
    :return:
    """
    path_to_buyable_data = 'data/pricer_using_reaxys_v2-chemicals_and_reaxys_v2-buyables.pkl'
    with open(path_to_buyable_data, 'rb') as f:
        buyable_molecules = pickle.load(f)

    try:
        return buyable_molecules[molecule]
    except:
        return 0


def _gaussian_wrapper(x, mu, sigma):
    if x < 3:
        return 1
    else:
        return np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


def gaussian_wrapper(x, mu=3, sigma=1):
    if isinstance(x, float):
        x = [x]

    return [_gaussian_wrapper(y, mu, sigma) for y in x]


def sa_score(smi):
    if isinstance(smi, str):
        smi = [smi]
    return [sascorer.calculateScore(Chem.MolFromSmiles(smiles)) for smiles in smi]


def get_synthesizability(molecule):
    # Check if the molecule is buyable first
    buyable = get_buyability(molecule)
    if buyable:
        return 1.0
    else:
        # If not buyable, then call the tree builder oracle
        HOST = 'http://askcos3.mit.edu'
        params = {
            'smiles': molecule,  # required
            # optional with defaults shown
            'max_depth': 5,
            'max_branching': 25,
            'expansion_time': 60,
            'max_ppg': 100,
            'template_count': 1000,
            'max_cum_prob': 0.999,
            'chemical_property_logic': 'none',
            'max_chemprop_c': 0,
            'max_chemprop_n': 0,
            'max_chemprop_o': 0,
            'max_chemprop_h': 0,
            'chemical_popularity_logic': 'none',
            'min_chempop_reactants': 5,
            'min_chempop_products': 5,
            'filter_threshold': 0.1,

            'return_first': 'true'  # default is false
        }

        for _ in range(15):
            resp = requests.get(HOST + '/api/treebuilder/', params=params)
            if 'error' not in resp.json().keys():
                break

        if 'error' not in resp.json().keys() or len(resp.json()['trees']) == 0:
            # No retrosynthetic pathway is found
            sa_score = sascorer.calculateScore(Chem.MolFromSmiles(molecule))
            return gaussian_wrapper(sa_score)
        else:
            # Retrosynthetic pathway is found
            return synthesizability_wrapper(resp.json())


def synthesizability_wrapper(json):
    num_path, status, depth, p_score, synthesizability, d_p = tree_analysis(json)
    return d_p


def tree_analysis(current):
    """
    Analise the result of tree builder
    Calculate: 1. Number of steps 2. \Pi plausibility 3. If find a path
    In case of celery error, all values are -1

    return:
        num_path = number of paths found
        status: Same as implemented in ASKCOS one
        num_step: number of steps
        p_score: \Pi plausibility
        synthesizability: binary code
    """
    if 'error' in current.keys():
        return -1, {}, -1, -1, -1

    num_path = len(current['trees'])
    if num_path != 0:
        current = [current['trees'][0]]
    else:
        current = []
    depth = 0
    p_score = 1
    status = {0: 1}
    while True:
        num_child = 0
        depth += 0.5
        temp = []
        for i, item in enumerate(current):
            num_child += len(item['children'])
            temp = temp + item['children']
        if num_child == 0:
            break

        if depth % 1 != 0:
            for sth in temp:
                p_score = p_score * sth['plausibility']

        status[depth] = num_child
        current = temp

    if len(status) > 1:
        synthesizability = 1
    else:
        synthesizability = 0
    return num_path, status, int(depth - 0.5), p_score * synthesizability, synthesizability, depth * (1 - 0.5 * p_score)


# def sc_score(smi):
#     scscorer = SCScorer()
#     scscorer.restore()
#
#     if isinstance(smi, str):
#         smi = [smi]
#
#     return [scscorer.apply(scscorer.smi_to_fp(smiles)) for smiles in smi]
