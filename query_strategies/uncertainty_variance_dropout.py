import numpy as np
import torch
from .strategy import Strategy
from dataset.data import MoleculeDataset


class UVarianceDropout(Strategy):

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		mol_unlabeled = MoleculeDataset(self.data[idxs_unlabeled])
		preds = self.predict_prob_dropout_split(mol_unlabeled)
		pred_var = torch.Tensor(preds.var(1))
		return idxs_unlabeled[pred_var.sort()[1][:n]]
