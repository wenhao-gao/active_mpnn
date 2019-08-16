import numpy as np
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from .strategy import Strategy
from dataset.data import MoleculeDataset


class MaxMin(Strategy):

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		if self.args.data_pool is not None:
			idxs_unlabeled = np.random.choice(idxs_unlabeled, self.args.data_pool, replace=False)

		embedding = self.get_embedding(MoleculeDataset(self.data[idxs_unlabeled]))

		def distij(i, j, data=embedding):
			return sum(np.sqrt(np.square(np.array(data[i]) - np.array(data[j]))))

		picker = MaxMinPicker()
		pickIndices = picker.LazyPick(distij, embedding.shape[0], n)

		return idxs_unlabeled[pickIndices]
