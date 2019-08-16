import numpy as np
from .strategy import Strategy


class RandomSampling(Strategy):

	def query(self, n):
		return np.random.choice(np.where(self.idxs_lb == 0)[0], n, replace=False)
