#!/usr/bin/env python
# encoding: utf-8
# File Name: mpnn.py
# Author: Joseph Xu
# Create Time: 2019/04/23 17:38
# TODO:

import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from Alchemy_dataset_3Ddis import TencentAlchemyDataset
from torch_geometric.utils import remove_self_loops

from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
import pandas as pd
from layers import RBFLayer

tmpdir=sys.argv[1]
bs=int(sys.argv[2])
num_epochs=int(sys.argv[3])
weight_loss = float(sys.argv[4])
model_id = int(sys.argv[5])
pdir= sys.argv[6]
lr_init=float(sys.argv[7])

in_cutoff = float(sys.argv[8])
in_gap = float(sys.argv[9])

class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data

if not os.path.exists(tmpdir):
    os.makedirs(tmpdir)

transform = T.Compose([Complete()])
train_dataset = TencentAlchemyDataset(root='data-bin', processed_dir=pdir, mode='dev', transform=transform).shuffle()
valid_dataset = TencentAlchemyDataset(root='data-bin', processed_dir=pdir, mode='valid', transform=transform)
print(train_dataset.data)
print(valid_dataset.data)

# bs=64

# test_loader = DataLoader(train_dataset, batch_size=bs)
valid_loader = DataLoader(valid_dataset, batch_size=bs)
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

class MPNN_rbf(torch.nn.Module):
    def __init__(self,
                 node_input_dim=15,
                 edge_input_dim=5,
                 output_dim=12,
                 node_hidden_dim=64,
                 edge_hidden_dim=128,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 cutoff = in_cutoff,
                 gap = in_gap):
        super(MPNN_rbf, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.rbflayer=RBFLayer(0, cutoff , gap)
        edge_network = nn.Sequential(
                nn.Linear(edge_input_dim+int(cutoff/gap)-1, edge_hidden_dim),
                nn.ReLU(),
                nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim)
                )
                
        self.conv = NNConv(node_hidden_dim, node_hidden_dim, edge_network, aggr='mean', root_weight=False)
        self.gru = nn.GRU(node_hidden_dim, node_hidden_dim)

        self.set2set = Set2Set(node_hidden_dim, processing_steps=num_step_set2set)
        self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        self.lin2 = nn.Linear(node_hidden_dim, output_dim)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)
        edge_attr_rbf=self.rbflayer(data.edge_attr)
        for i in range(self.num_step_message_passing):
            m = F.relu(self.conv(out, data.edge_index, edge_attr_rbf))
            out1, h = self.gru(m.unsqueeze(0), h)
            out = out1.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)

        # out_edge=self.edge_network_dec(self.edge_network_enc(data.edge_attr))
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MPNN_rbf(node_input_dim=train_dataset.num_node_features, edge_input_dim=train_dataset.num_edge_features).to(device)
print(model)
print(f"Total parameters: {sum([p.numel() for p in model.parameters()])}")
if model_id>0:
    model.load_state_dict(torch.load(f"{tmpdir}/model_epoch_{model_id}.pkl"))
    print(f"Loading the pretrained model: {tmpdir}/model_epoch_{model_id}.pkl")

optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.96, patience=5, min_lr=0.00001)
MAE_fn = nn.L1Loss()

def train(epoch):
    model.train()
    loss_all = 0
    mae_loss_all=0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        y_model = model(data)
        loss_y = F.mse_loss(y_model, data.y)
        mae_loss=MAE_fn(y_model, data.y)
        # edge_loss=F.mse_loss(edge_model, data.edge_attr)
        loss= loss_y
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        mae_loss_all += mae_loss.item() * data.num_graphs
        optimizer.step()

    torch.save(model.state_dict(), f"{tmpdir}/model_epoch_{epoch}.pkl")
    return loss_all / len(train_loader.dataset), mae_loss_all/ len(train_loader.dataset)


def test(loader):
    model.eval()
    error=0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            y_pred = model(data)
            error += (y_pred - data.y).abs().sum().item()
            del y_pred
        # for i in range(len(data.y)):
        #     targets[data.y[i].item()] = y_pred[i].tolist()
    return  error/len(loader.dataset.data.y.view(-1))

# num_epochs = 301
best_val_error = None
best_val_id=None

import time
print("training...")
for epoch in range(model_id+1, num_epochs+1):
    lr = scheduler.optimizer.param_groups[0]['lr']

    st=time.time()
    loss, train_error = train(epoch)
    val_error=test(valid_loader)
    scheduler.step(val_error)

    if best_val_error is None or val_error <= best_val_error:
        best_val_error = val_error
        best_val_id=epoch
    # print('Epoch: {:03d}, Loss: {:.7f}'.format(epoch, loss))
    print('Time: {:.2f}s, Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, MAE: {.7f}, Validation MAE: {:.7f}'
          .format(time.time()-st, epoch, lr, loss, train_error, val_error))

print(f"ID of the best model: {best_val_id}; Val MAE: {best_val_error}")
# print("The best MAE: {:.7f}".format(best_val_error))
# targets = test(valid_loader)
# df_targets = pd.DataFrame.from_dict(targets, orient="index", columns=['property_%d' % x for x in range(12)])
# df_targets.sort_index(inplace=True)
# df_targets.to_csv('targets.csv', index_label='gdb_idx')
