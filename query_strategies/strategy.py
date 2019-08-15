import os
import numpy as np
import torch.nn.functional as F
from argparse import Namespace
import logging
from typing import List, Union

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import trange
from dataset.data import MoleculeDataset


def compute_mse(y1, y2):
    return np.mean(np.square(y1 - y2))


def compute_mae(y1, y2):
    return np.mean(np.absolute(y1 - y2))


def compute_maxae(y1, y2):
    return np.max(np.absolute(y1 - y2))


class Strategy:
    """A base active learning query strategy class
    The query function should be overwritten by specific method
    """

    def __init__(self,
                 data: Union[MoleculeDataset, List[MoleculeDataset]],
                 idxs_lb: List,
                 net: nn.Module,
                 optimizer: Optimizer,
                 lr_schedule: _LRScheduler,
                 args: Namespace,
                 logger: logging.Logger = None,
                 writer: SummaryWriter = None):
        """
        Initialize a strategy, with dataset and network.

        :param data: training data
        :param idxs_lb: labeled index
        :param net: network
        :param optimizer: optimizer
        :param lr_schedule: learning rate decay scheduler
        :param args: argument
        :param logger: result logger
        :param writer: tensorboard writer
        """
        self.data = data
        self.idxs_lb = idxs_lb
        self.net = net
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.args = args
        self.logger = logger
        self.writer = writer
        self.loss_func = F.smooth_l1_loss
        self.n_pool = len(data)
        self.n_drop = args.n_drop

        # GPU setting
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.net = self.net.to(self.device)

    def query(self, n):
        pass

    def update(self, idxs_lb):
        """Update labeled index after query"""
        self.idxs_lb = idxs_lb

    def _train(self,
               epoch: int,
               data: Union[MoleculeDataset, List[MoleculeDataset]],
               n_iter: int) -> int:
        """
        Trains a model for an epoch.
        """
        debug = self.logger.debug if self.logger is not None else print

        debug(f'Running epoch: {epoch}')

        self.net.train()

        data.shuffle()
        loss_sum, iter_count = 0, 0
        num_iters = len(data) // self.args.batch_size * self.args.batch_size
        iter_size = self.args.batch_size

        for i in trange(0, num_iters, iter_size):
            # Prepare batch
            if i + self.args.batch_size > len(data):
                break
            mol_batch = MoleculeDataset(data[i:i + self.args.batch_size])
            smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
            batch = smiles_batch
            mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
            targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

            if next(self.net.parameters()).is_cuda:
                mask, targets = mask.cuda(), targets.cuda()

            class_weights = torch.ones(targets.shape)

            if self.use_cuda:
                class_weights = class_weights.cuda()

            # Run model
            self.net.zero_grad()
            preds, e = self.net(batch, features_batch)

            loss = self.loss_func(preds, targets) * class_weights * mask
            loss = loss.sum() / mask.sum()

            loss_sum += loss.item()
            iter_count += len(mol_batch)

            loss.backward()
            self.optimizer.step()

            if (n_iter // self.args.batch_size) % self.args.learning_rate_decay_steps == 0:
                self.lr_schedule.step()

            n_iter += len(mol_batch)

            # Log and/or add to tensorboard
            if (n_iter // self.args.batch_size) % self.args.log_frequency == 0:
                lrs = self.lr_schedule.get_lr()
                loss_avg = loss_sum / iter_count
                loss_sum, iter_count = 0, 0

                lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
                debug(f'Loss = {loss_avg:.4e}, {lrs_str}')

                if self.writer is not None:
                    self.writer.add_scalar('train_loss', loss_avg, n_iter)
                    # for i, lr in enumerate(lrs):
                    #     self.writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

        return n_iter

    def train(self, n_iter, n_epoch=None):
        if n_epoch is None:
            n_epoch = self.args.epoch

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        data = MoleculeDataset(self.data[idxs_train])

        for epoch in range(1, n_epoch+1):
            n_iter = self._train(epoch, data, n_iter)

        return n_iter

    def predict(self, data, scaler=None):
        self.net.eval()

        preds = []

        num_iters, iter_step = len(data), self.args.batch_size

        for i in range(0, num_iters, iter_step):
            # Prepare batch
            mol_batch = MoleculeDataset(data[i:i + self.args.batch_size])
            smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

            # Run model
            batch = smiles_batch

            with torch.no_grad():
                batch_preds, e = self.net(batch, features_batch)

            batch_preds = batch_preds.data.cpu().numpy()

            # Inverse scale if regression
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)

            # Collect vectors
            batch_preds = batch_preds.tolist()
            preds.extend(batch_preds)

        return preds

    def evaluate(self, data):
        preds = np.array(self.predict(data))
        targets = np.array(data.targets())
        return compute_mse(preds, targets), compute_mae(preds, targets), compute_maxae(preds, targets)

    def predict_prob_dropout_split(self, data, scaler=None):
        self.net.train()

        preds = []

        num_iters, iter_step = len(data), self.args.batch_size

        for i in range(0, num_iters, iter_step):
            # Prepare batch
            mol_batch = MoleculeDataset(data[i:i + self.args.batch_size])
            smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

            # Run model
            batch = smiles_batch

            batch_preds = []
            with torch.no_grad():
                for i in range(self.n_drop):
                    batch_pred, e = self.net(batch, features_batch)
                    batch_preds.append(batch_pred.data.cpu().numpy())

            # Inverse scale if regression
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)

            # Collect vectors
            batch_preds = np.hstack(batch_preds).tolist()
            preds.extend(batch_preds)

        return np.array(preds)

    def get_embedding(self, data):
        self.net.eval()

        embedding = []

        num_iters, iter_step = len(data), self.args.batch_size

        for i in range(0, num_iters, iter_step):
            # Prepare batch
            mol_batch = MoleculeDataset(data[i:i + self.args.batch_size])
            smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

            # Run model
            batch = smiles_batch

            with torch.no_grad():
                preds, latent = self.net(batch, features_batch)

            latent = latent.data.cpu().numpy()

            # Collect vectors
            latent = latent.tolist()
            embedding.extend(latent)

        return np.array(embedding)

    def save_net(self, rd, name='model'):
        if not os.path.exists(self.args.log_path):
            os.makedirs(self.args.log_path)

        model_name = name + '_checkpoint_' + str(rd) + '.pth'
        torch.save(self.net.state_dict(), os.path.join(self.args.log_path, model_name))

