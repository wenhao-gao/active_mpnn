import os
import numpy as np
from dataset.dataset import get_dataset, get_handler
from model.nn_utils import initialize_weights
from model.get_models import get_net
from oracle.get_oracles import get_oracle
import torch
import torch.optim as optim
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                                LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                                KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
                                AdversarialBIM, AdversarialDeepFool, ActiveLearningByLearning

# parameters
NUM_INIT_LB = 1000
DATA_NAME = 'regression'


class N:
    def __init__(self):
        self.task = 'test'
        self.seed = 1
        self.round = 3
        self.query = 128
        self.log_path = 'checkpoints/'
        self.init_data = 'data/sheridan_train.csv'
        self.test_data = 'data/sheridan_test.csv'
        self.pool_data = 'data/chembl250k.csv'
        self.mol_col = 'SMILES'
        self.mol_prop = 'sa_score'
        self.network = 'mpnn'
        self.param = None
        self.num_bootstrap_heads = 1
        self.hidden_size = 300
        self.bias = False
        self.depth = 3
        self.ffn_hidden_size = 300
        self.ffn_num_layers = 2
        self.atom_messages = False
        self.feature_only = False
        self.use_input_features = True
        self.features_dim = 1
        self.dropout = 0.5
        self.activation = 'ReLU'
        self.strategy = 'random'
        self.oracle = 'test'
        self.learning_rate = 0.01
        self.adam_beta_1 = 0.9
        self.adam_beta_2 = 0.999
        self.epoch = 10
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.transform = None
        self.loader_tr_args = {'batch_size': 64, 'num_workers': 1}
        self.loader_te_args = {'batch_size': 1000, 'num_workers': 1}


def mse(y1, y2):
    return np.mean(np.square(y1 - y2))


args = N()

X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME)
X_tr = X_tr[:10000]
Y_tr = Y_tr[:10000]

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)
print('number of labeled pool: {}'.format(NUM_INIT_LB))
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
print('number of testing pool: {}'.format(n_test))

# generate initial labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = np.arange(n_pool)
np.random.shuffle(idxs_tmp)
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

net = get_net(DATA_NAME)
net = net()
initialize_weights(net)
if args.param is not None:
    net.load_state_dict(torch.load(args.param))
handler = get_handler(DATA_NAME)
optimizer = optim.SGD(net.parameters(),
                      lr = args.learning_rate,
                      momentum = 0.5
                     )
oracle = get_oracle(args)
strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, optimizer, args)

print(DATA_NAME)
print('SEED {}'.format(args.seed))
print(type(strategy).__name__)
losses = []

strategy.train()
pred = strategy.predict(X_te, Y_te)
loss = mse(pred.cpu().numpy(), Y_te.cpu().numpy())
losses.append(loss)
print('Round 0\ntesting MSE {}'.format(loss))

for rd in range(1, args.round + 1):
    print('Round {}'.format(rd))

    # query
    q_idxs = strategy.query(args.query)
    oracle(X_tr, Y_tr, idxs_lb, q_idxs)

    # update
    strategy.update(idxs_lb)
    strategy.train()

    # round accuracy
    pred = strategy.predict(X_te, Y_te)
    loss = mse(pred.cpu().numpy(), Y_te.cpu().numpy())
    losses.append(loss)
    print('Round {}\ntesting MSE {}'.format(rd, loss))

    strategy.save_net(rd)

# print results
print('SEED {}'.format(args.seed))
print(type(strategy).__name__)
print(losses)
