# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn

class RBFLayer(nn.Module):
    """
    Radial basis functions Layer.
    e(d) = exp(- gamma * ||d - mu_k||^2)
    default settings:
        gamma = 10
        0 <= mu_k <= 30 for k=1~300
    """
    def __init__(self, low=0, high=10, gap=0.5, dim=1):
        super().__init__()
        self._low = low
        self._high = high
        self._gap = gap
        self._dim = dim

        self._n_centers = int(np.ceil((high - low) / gap))
        centers = np.linspace(low, high, self._n_centers)
        self.centers = torch.tensor(centers, dtype=torch.float, requires_grad=False)
        self.centers = nn.Parameter(self.centers, requires_grad=False)
        self._fan_out = self._dim * self._n_centers

        self._gap = centers[1] - centers[0]

    def dis2rbf(self, edges_attr):
        dist = edges_attr
        radial = dist - self.centers.view(1, -1).repeat(dist.size()[0], 1)
        coef = -1 / self._gap
        rbf = torch.exp(coef * (radial ** 2))
        # print(rbf.size())
        return rbf

    def forward(self, edges_attr):
        """Convert distance scalar to rbf vector"""
        dis_rbf = self.dis2rbf(edges_attr[:, -1:])
        out = torch.cat([edges_attr[:, :-1], dis_rbf], 1)
        # print(out.size())
        return out