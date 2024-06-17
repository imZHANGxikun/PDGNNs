import random
import numpy as np
from numpy.random import choice
import torch
import torch.nn as nn
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CoraGraphDataset, CoraFullDataset, register_data_args, RedditDataset


class random_select(nn.Module):
    def __init__(self,args=None):
        super(random_select, self).__init__()
    def forward(self, ids_per_cls_train, budget, neighbor_agg_model=None, graph=None, topo_vecs=None, max_ratio_per_cls = 1.0):
        store_ids = []
        for i,ids in enumerate(ids_per_cls_train):
            budget_ = min(budget, int(max_ratio_per_cls * len(ids))) if isinstance(budget, int) else int(budget * len(ids))
            store_ids.extend(random.sample(ids, budget_))
        return store_ids

class cover_max_select_01(nn.Module):
    # in this version, the nodes are simultaneously sampled according to their covered size, instead of repeatedly increasing single nodes and evaluating the covered areas
    def __init__(self,args=None):
        super().__init__()

    def forward(self, ids_per_cls, budget, neighbor_agg_model=None, graph=None, topo_vecs=None, max_ratio_per_cls = 1.0):
        n_nodes = graph.dstdata['label'].shape[0]
        d = torch.ones(n_nodes)
        indicators = torch.diag(d).bool().cuda(graph.device)
        cover_per_node = neighbor_agg_model(graph, indicators).bool()
        return self.select(cover_per_node, budget, ids_per_cls, n_nodes)

    def select(self, cover_per_node, budget, ids_per_cls, n_nodes):
        ids_selected = []
        for ids in ids_per_cls:
            judge = cover_per_node.bool().sum(1)[ids]
            judge_prob = (judge/judge.sum()).cpu().double()
            judge_prob /= judge_prob.sum()
            ids_selec = choice(ids, min(budget, len(ids)),replace=False, p=judge_prob)
            ids_selected.extend(ids_selec)
        return ids_selected


class cover_max_select_02(nn.Module):
    # compared to 01, this version use degree as surrogate of the coverage
    def __init__(self,args=None):
        super().__init__()
        self.covered_nodes = []
        self.covered_nodes_random = []
        self.covered_nodes_herd = []
        self.random_select = random_select()

    def forward(self, ids_per_cls, budget, neighbor_agg_model=None, graph=None, topo_vecs=None, max_ratio_per_cls = 1.0, batch_size=10):
        n_nodes = graph.dstdata['label'].shape[0]

        ind = graph.in_degrees()
        selected_ids = self.select(ind, budget, ids_per_cls, n_nodes)
        return selected_ids

    def select(self, cover_per_node, budget, ids_per_cls, n_nodes):
        ids_selected = []
        for ids in ids_per_cls:
            judge = cover_per_node[ids].half()
            ids_of_ids = torch.multinomial(judge, min(budget, len(ids)), replacement=False).tolist()
            ids_selec = torch.tensor(ids)[ids_of_ids].tolist()
            ids_selected.extend(ids_selec)
        return ids_selected

