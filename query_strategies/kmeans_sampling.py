import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans
from dataset.data import MoleculeDataset


class KMeansSampling(Strategy):

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		if self.args.data_pool is not None:
			idxs_unlabeled = np.random.choice(idxs_unlabeled, self.args.data_pool, replace=False)

		embedding = self.get_embedding(MoleculeDataset(self.data[idxs_unlabeled]))

		cluster_learner = KMeans(n_clusters=n)
		cluster_learner.fit(embedding)
		
		cluster_idxs = cluster_learner.predict(embedding)
		centers = cluster_learner.cluster_centers_[cluster_idxs]
		dis = (embedding - centers)**2
		dis = dis.sum(axis=1)
		q_idxs = np.array(
			[
				np.arange(embedding.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()]
				for i in range(n)
			]
		)

		return idxs_unlabeled[q_idxs]
