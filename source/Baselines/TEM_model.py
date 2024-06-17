
from .TEM_utils import *

sampler_types = {'random_select': random_select, 'cover_max_select_01':cover_max_select_01, 'cover_max_select_02':cover_max_select_02}
class NET(torch.nn.Module):
    def __init__(self,
                 model,
                 task_manager,
                 args):
        super(NET, self).__init__()

        self.task_manager = task_manager

        # setup network
        self.net = model

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.functional.cross_entropy

        # setup memories
        self.current_task = -1
        self.sampler = sampler_types[args.tem_args['sampler']](args) if not isinstance(args.tem_args['sampler'], list) else sampler_types[args.tem_args['sampler'][0]](args)
        self.TEM_vecs = torch.tensor([]).cuda(args.gpu)
        self.TEM_labels = torch.tensor([]).long().cuda(args.gpu)
        self.budget = args.tem_args['budget']

    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, args, g, features, labels_all, t, train_ids, ids_per_cls, dataset):
        self.net.train()
        labels = labels_all[train_ids]
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        topo_vecs = self.net.neighbor_agg(g, features)
        input_feat_concat = torch.cat([topo_vecs[train_ids], self.TEM_vecs], dim=0)
        labels_concat = torch.cat([labels, self.TEM_labels], dim=0)
        if t!=self.current_task:
            self.current_task = t
            selected_ids = self.sampler(ids_per_cls_train, self.budget, neighbor_agg_model=self.net.neighbor_agg, graph=g, topo_vecs=topo_vecs)
            self.TEM_vecs = torch.cat([self.TEM_vecs, topo_vecs[selected_ids]], dim=0)
            self.TEM_labels = torch.cat([self.TEM_labels, labels_all[selected_ids]], dim=0)
        if args.cls_balance:
            n_per_cls = [(labels_concat == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        output, _ = self.net.feat_trans(g, input_feat_concat)
        
        if args.classifier_increase:
            loss = self.ce(output[:,offset1:offset2], labels_concat, weight=loss_w_[offset1: offset2])
        else:
            loss = self.ce(output, labels_concat, weight=loss_w_)

        loss.backward()
        self.opt.step()

    def observe_task_IL(self, args, g, features, labels_all, t, train_ids, ids_per_cls, dataset):
        if not isinstance(self.TEM_vecs, list):
            self.TEM_vecs = []
            self.TEM_labels = []
        self.net.train()
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
        topo_vecs = self.net.neighbor_agg(g, features)
        input_feat_concat = topo_vecs[train_ids]
        labels_concat = labels_all[train_ids]
        if t!=self.current_task:
            self.current_task = t
            selected_ids = self.sampler(ids_per_cls_train, self.budget, neighbor_agg_model=self.net.neighbor_agg, graph=g, topo_vecs=topo_vecs)
            self.TEM_vecs.append(topo_vecs[selected_ids])
            self.TEM_labels.append(labels_all[selected_ids])
        if args.cls_balance:
            n_per_cls = [(labels_concat == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
        output, _ = self.net.feat_trans(g, input_feat_concat)
        loss = self.ce(output[:, offset1:offset2], labels_concat-offset1, weight=loss_w_[offset1: offset2])

        # memory replay
        if t>0:
            for oldt in range(t):
                o1, o2 = self.task_manager.get_label_offset(oldt-1)[1], self.task_manager.get_label_offset(oldt)[1]
                replay_sses = self.TEM_vecs[oldt]
                replay_labels = self.TEM_labels[oldt]
                if args.cls_balance:
                    n_per_cls = [(replay_labels == j).sum() for j in range(args.n_cls)]
                    loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
                else:
                    loss_w_ = [1. for i in range(args.n_cls)]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                logits_replay , _ = self.net.feat_trans(g, replay_sses)
                loss_aux = self.ce(logits_replay[:,o1:o2], replay_labels-o1, weight=loss_w_[offset1: offset2])
                loss = loss + loss_aux

        loss.backward()
        self.opt.step()
