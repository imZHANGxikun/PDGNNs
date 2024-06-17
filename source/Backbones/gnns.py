from .gnnconv import GATConv, GCNLayer, GINConv
from .layers import PairNorm
import torch.nn.init as init
from .utils import *
from dgl.base import DGLError
from dgl.nn.pytorch import edge_softmax
import dgl.function as fn
from dgl.nn.pytorch.conv import APPNPConv
import torch as th
linear_choices = {'nn.Linear':nn.Linear, 'Linear_IL':Linear_IL}

class GIN(nn.Module):
    def __init__(self,
                 args,):
        super(GIN, self).__init__()
        dims = [args.d_data] + args.GIN_args['h_dims'] + [args.n_cls]
        self.dropout = args.GIN_args['dropout']
        self.gat_layers = nn.ModuleList()
        for l in range(len(dims)-1):
            lin = torch.nn.Linear(dims[l], dims[l+1])
            self.gat_layers.append(GINConv(lin, 'sum'))


    def forward(self, g, features):
        e_list = []
        h, e = self.gat_layers[0](g, features)
        x = F.relu(h)
        logits, e = self.gat_layers[1](g, x)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, features):
        e_list = []
        h, e = self.gat_layers[0].forward_batch(blocks[0], features)
        x = F.relu(h)
        logits, e = self.gat_layers[1].forward_batch(blocks[1], x)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

class GCN(nn.Module):
    def __init__(self,
                 args):
        super(GCN, self).__init__()
        dims = [args.d_data] + args.GCN_args['h_dims'] + [args.n_cls]
        self.dropout = args.GCN_args['dropout']
        self.gat_layers = nn.ModuleList()
        for l in range(len(dims)-1):
            self.gat_layers.append(GCNLayer(dims[l], dims[l+1]))

    def forward(self, g, features):
        e_list = []
        h = features
        for layer in self.gat_layers[:-1]:
            h, e = layer(g, h)
            h = F.relu(h)
            e_list = e_list + e
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits, e = self.gat_layers[-1](g, h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, features):
        e_list = []
        h = features
        for i,layer in enumerate(self.gat_layers[:-1]):
            h, e = layer.forward_batch(blocks[i], h)
            h = F.relu(h)
            e_list = e_list + e
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits, e = self.gat_layers[-1].forward_batch(blocks[-1], h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list


    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

class GAT(nn.Module):
    def __init__(self,
                 args,
                 heads,
                 activation):
        super(GAT, self).__init__()
        #self.g = g
        self.num_layers = args.GAT_args['num_layers']
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            args.d_data, args.GAT_args['num_hidden'], heads[0],
            args.GAT_args['feat_drop'], args.GAT_args['attn_drop'], args.GAT_args['negative_slope'], False, None))
        # self.norm_layers.append(nn.BatchNorm1d(num_hidden*heads[0]))
        self.norm_layers.append(PairNorm())
        
        # hidden layers
        for l in range(1, args.GAT_args['num_layers']):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                args.GAT_args['num_hidden'] * heads[l-1], args.GAT_args['num_hidden'], heads[l],
                args.GAT_args['feat_drop'], args.GAT_args['attn_drop'], args.GAT_args['negative_slope'], args.GAT_args['residual'], self.activation))
            # self.norm_layers.append(nn.BatchNorm1d(num_hidden*heads[l]))
            self.norm_layers.append(PairNorm())
        # output projection

        self.gat_layers.append(GATConv(
            args.GAT_args['num_hidden'] * heads[-2], args.n_cls, heads[-1],
            args.GAT_args['feat_drop'], args.GAT_args['attn_drop'], args.GAT_args['negative_slope'], args.GAT_args['residual'], None))

    def forward(self, g, inputs, save_logit_name = None):
        h = inputs
        e_list = []
        for l in range(self.num_layers):
            h, e = self.gat_layers[l](g, h)
            h = h.flatten(1)
            h = self.activation(h)
            e_list = e_list + e
        # store for ergnn
        self.second_last_h = h
        # output projection
        logits, e = self.gat_layers[-1](g, h)
        #self.second_last_h = logits if len(self.gat_layers) == 1 else h
        logits = logits.mean(1)
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, features):
        e_list = []
        h = features
        for i,layer in enumerate(self.gat_layers[:-1]):
            h, e = layer.forward_batch(blocks[i], h)
            h = h.flatten(1)
            h = self.activation(h)
            e_list = e_list + e
        logits, e = self.gat_layers[-1].forward_batch(blocks[-1], h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        logits = logits.mean(1)
        e_list = e_list + e
        return logits, e_list


    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()


class SGC_Agg(nn.Module):
    # only the neighborhood aggregation of SGC
    def __init__(self, k=1, cached=False, norm=None, allow_zero_in_degree=False):
        super().__init__()
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute Simplifying Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

        Note
        ----
        If ``cache`` is set to True, ``feat`` and ``graph`` should not change during
        training, or you will get wrong results.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X
                for _ in range(self._k):
                    feat = feat * norm
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm

                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat
            # return self.fc(feat)
            return feat

    def forward_batch(self, blocks, feat):
        r"""

        Description
        -----------
        Compute Simplifying Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

        Note
        ----
        If ``cache`` is set to True, ``feat`` and ``graph`` should not change during
        training, or you will get wrong results.
        """
        if self._k != len(blocks):
            raise DGLError('The depth of the dataloader sampler is incompatible with the depth of SGC')
        for block in blocks:
            with block.local_scope():
                if not self._allow_zero_in_degree:
                    if (block.in_degrees() == 0).any():
                        raise DGLError('There are 0-in-degree nodes in the graph, '
                                       'output for those nodes will be invalid. '
                                       'This is harmful for some applications, '
                                       'causing silent performance regression. '
                                       'Adding self-loop on the input graph by '
                                       'calling `g = dgl.add_self_loop(g)` will resolve '
                                       'the issue. Setting ``allow_zero_in_degree`` '
                                       'to be `True` when constructing this module will '
                                       'suppress the check and let the code run.')

                if self._cached_h is not None:
                    feat = self._cached_h
                else:
                    # compute normalization
                    degs = block.out_degrees().float().clamp(min=1)
                    norm = th.pow(degs, -0.5)
                    norm = norm.to(feat.device).unsqueeze(1)
                    # compute (D^-1 A^k D)^k X
                    feat = feat * norm
                    block.srcdata['h'] = feat
                    block.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                    feat = block.dstdata.pop('h')
                    degs = block.in_degrees().float().clamp(min=1)
                    norm = th.pow(degs, -0.5)
                    norm = norm.to(feat.device).unsqueeze(1)
                    feat = feat * norm

        with blocks[-1].local_scope():
            if self.norm is not None:
                feat = self.norm(feat)

            # cache feature
            if self._cached:
                self._cached_h = feat

        # return self.fc(feat)
        return feat

class SGC(nn.Module):
    def __init__(self, args):
        super(SGC, self).__init__()
        linear_layer = linear_choices[args.SGC_args['linear']]
        if args.method == 'twp':
            self.twp=True
        else:
            self.twp=False
        self.bn = args.SGC_args['batch_norm']
        self.dropout = args.SGC_args['dropout']
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.gpu = args.gpu
        self.neighbor_agg = SGC_Agg(k=args.SGC_args['k'])
        self.feat_trans_layers = nn.ModuleList()
        if self.bn:
            self.bns = nn.ModuleList()
        h_dims = args.SGC_args['h_dims']
        if len(h_dims) > 0:
            self.feat_trans_layers.append(linear_layer(args.d_data, h_dims[0], bias=args.SGC_args['linear_bias']))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(h_dims[0]))
            for i in range(len(h_dims) - 1):
                self.feat_trans_layers.append(linear_layer(h_dims[i], h_dims[i + 1], bias=args.SGC_args['linear_bias']))
                if self.bn:
                    self.bns.append(nn.BatchNorm1d(h_dims[i + 1]))
            self.feat_trans_layers.append(linear_layer(h_dims[-1], args.n_cls, bias=args.SGC_args['linear_bias']))
        elif len(h_dims) == 0:
            self.feat_trans_layers.append(linear_layer(args.d_data, args.n_cls, bias=args.SGC_args['linear_bias']))
        else:
            raise ValueError('no valid MLP dims are given')

    def forward(self, graph, x, twp=False, tasks=None):
        graph = graph.local_var().to('cuda:{}'.format(self.gpu))
        e_list = []
        x = self.neighbor_agg(graph, x)
        logits, e = self.feat_trans(graph, x, twp=twp, cls=tasks)
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, x, twp=False, tasks=None):
        #graph = graph.local_var().to('cuda:{}'.format(self.gpu))
        e_list = []
        x = self.neighbor_agg.forward_batch(blocks, x)
        logits, e = self.feat_trans(blocks[0], x, twp=twp, cls=tasks)
        e_list = e_list + e
        return logits, e_list

    def feat_trans(self, graph, x, twp=False, cls=None):
        for i, layer in enumerate(self.feat_trans_layers[:-1]):
            x = layer(x)
            if self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.feat_trans_layers[-1](x)

        self.second_last_h = x

        mask = torch.zeros(x.shape[-1], device=x.get_device())
        if cls is not None:
            mask[cls] = 1.
        else:
            mask[:] = 1.
        x = x * mask
        # for twp
        elist = []
        if self.twp:
            graph.srcdata['h'] = x
            graph.apply_edges(
                lambda edges: {'e': torch.sum((torch.mul(edges.src['h'], torch.tanh(edges.dst['h']))), 1)})
            e = self.leaky_relu(graph.edata.pop('e'))
            e_soft = edge_softmax(graph, e)

            elist.append(e_soft)

        return x, elist
        #return x.log_softmax(dim=-1), elist
    def reset_params(self):
        for layer in self.feat_trans_layers:
            layer.reset_parameters()

class CustomDecoupledAPPNP(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.method == 'twp':
            self.twp=True
        else:
            self.twp=False
        self.gpu = args.gpu
        ags = args.CustomDecoupledAPPNP_args
        h_dims, dropout, bias, k, alpha, self.bn = ags['h_dims'], ags['dropout'], ags['bias'], ags['k'], ags['alpha'], ags['batch_norm']
        mlp_dims = [args.d_data] + h_dims + [args.n_cls]

        linear_layer = linear_choices[ags['linear']]

        fcs = []
        for i in range(1, len(mlp_dims)):
            fcs.append(linear_layer(mlp_dims[i - 1], mlp_dims[i], bias=bias))
        self.fcs = nn.ModuleList(fcs)
        self.reg_params = list(self.fcs[0].parameters())
        self.dropout = nn.Dropout(p=dropout)
        self.act_fn = nn.ReLU()
        self.neighbor_agg = APPNPConv(k, alpha)

    def feat_trans(self, graph, x, twp=False, cls=None):
        for fc in self.fcs[0:-1]:
            x = self.act_fn(fc(self.dropout(x)))
        x = self.fcs[-1](self.dropout(x))
        self.second_last_h = x
        # for twp
        elist = []
        if self.twp:
            graph.srcdata['h'] = x
            graph.apply_edges(
                lambda edges: {'e': torch.sum((torch.mul(edges.src['h'], torch.tanh(edges.dst['h']))), 1)})
            e = self.leaky_relu(graph.edata.pop('e'))
            e_soft = edge_softmax(graph, e)

            elist.append(e_soft)

        return x, elist

    def forward(self, graph, x, twp=False, tasks=None):
        graph = graph.local_var().to('cuda:{}'.format(self.gpu))
        e_list = []
        x = self.neighbor_agg(graph, x)
        logits, e = self.feat_trans(graph, x, twp=twp, cls=tasks)
        e_list = e_list + e
        return logits, e_list

    def reset_params(self):
        for layer in self.fcs:
            layer.reset_parameters()

class CustomS2GC_Agg(nn.Module):
    # only the neighborhood aggregation of S2GC
    def __init__(self,
                 alpha=0.05,
                 k=1,
                 cached=False,
                 norm=None,
                 allow_zero_in_degree=False):
        super().__init__()
        self._cached = cached
        self._cached_h = None
        self._alpha = alpha
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)
                # compute SUM_k(D^-1 A^k D)^k X
                feat_ori = copy.deepcopy(feat)
                feat_sum_k = 0
                for _ in range(self._k):
                    feat = feat * norm
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'),
                                     fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm
                    feat_sum_k+=feat
                feat = ((1.-self._alpha)/self._k)*feat_sum_k + self._alpha*feat_ori

                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat
            #return self.fc(feat)
            return feat

class CustomDecoupledS2GC(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.method == 'twp':
            self.twp=True
        else:
            self.twp=False
        self.gpu = args.gpu
        ags = args.CustomDecoupledS2GC_args
        linear_layer = linear_choices[ags['linear']]
        h_dims, self.dropout, bias, k, alpha, self.bn = ags['h_dims'], ags['dropout'], ags['bias'], ags['k'], ags['alpha'], ags['batch_norm']
        self.neighbor_agg = CustomS2GC_Agg(alpha=alpha, k=k)
        self.feat_trans_layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        dims = [args.d_data] + h_dims + [args.n_cls]

        fcs = []
        for i in range(1, len(dims)-1):
            fcs.append(linear_layer(dims[i - 1], dims[i], bias=bias))
            self.bns.append(nn.BatchNorm1d(dims[i]))
        fcs.append(linear_layer(dims[- 2], dims[-1], bias=bias))
        self.feat_trans_layers = nn.ModuleList(fcs)

    def feat_trans(self, graph, x, twp=False, cls=None):
        for i, layer in enumerate(self.feat_trans_layers[:-1]):
            x = layer(x)
            if self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.feat_trans_layers[-1](x)

        self.second_last_h = x

        mask = torch.zeros(x.shape[-1], device=x.get_device())
        if cls is not None:
            mask[cls] = 1.
        else:
            mask[:] = 1.
        x = x * mask

        # for twp
        elist = []
        if self.twp:
            graph.srcdata['h'] = x
            graph.apply_edges(
                lambda edges: {'e': torch.sum((torch.mul(edges.src['h'], torch.tanh(edges.dst['h']))), 1)})
            e = self.leaky_relu(graph.edata.pop('e'))
            e_soft = edge_softmax(graph, e)

            elist.append(e_soft)

        return x, elist

    def forward(self, graph, x, twp=False, tasks=None):
        graph = graph.local_var().to('cuda:{}'.format(self.gpu))
        e_list = []
        x = self.neighbor_agg(graph, x)
        logits, e = self.feat_trans(graph, x, twp=twp, cls=tasks)
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, x, tasks=None):
        x = self.neighbor_agg(blocks[-1], x)
        x = self.feat_trans(x, tasks)
        return x

    def reset_params(self):
        for layer in self.feat_trans_layers:
            layer.reset_parameters()

class CustomSGC_Agg(nn.Module):
    # only the neighborhood aggregation of SGC
    def __init__(self, k=1, cached=False, norm=None, allow_zero_in_degree=False):
        super().__init__()
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute Simplifying Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

        Note
        ----
        If ``cache`` is set to True, ``feat`` and ``graph`` should not change during
        training, or you will get wrong results.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X
                for _ in range(self._k):
                    feat = feat * norm
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm

                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat
            # return self.fc(feat)
            return feat

class CustomDecoupledSGC(nn.Module):
    def __init__(self, args):
        super(CustomDecoupledSGC, self).__init__()
        linear_layer = linear_choices[args.CustomDecoupledSGC_args['linear']]
        if args.method == 'twp':
            self.twp=True
        else:
            self.twp=False
        self.bn = args.CustomDecoupledSGC_args['batch_norm']
        self.dropout = args.CustomDecoupledSGC_args['dropout']
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.gpu = args.gpu
        self.neighbor_agg = CustomSGC_Agg(k=args.CustomDecoupledSGC_args['k'])
        self.feat_trans_layers = nn.ModuleList()
        if self.bn:
            self.bns = nn.ModuleList()
        h_dims = args.CustomDecoupledSGC_args['h_dims']
        if len(h_dims) > 0:
            self.feat_trans_layers.append(linear_layer(args.d_data, h_dims[0], bias=args.CustomDecoupledSGC_args['linear_bias']))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(h_dims[0]))
            for i in range(len(h_dims) - 1):
                self.feat_trans_layers.append(linear_layer(h_dims[i], h_dims[i + 1], bias=args.CustomDecoupledSGC_args['linear_bias']))
                if self.bn:
                    self.bns.append(nn.BatchNorm1d(h_dims[i + 1]))
            self.feat_trans_layers.append(linear_layer(h_dims[-1], args.n_cls, bias=args.CustomDecoupledSGC_args['linear_bias']))
        elif len(h_dims) == 0:
            self.feat_trans_layers.append(linear_layer(args.d_data, args.n_cls, bias=args.CustomDecoupledSGC_args['linear_bias']))
        else:
            raise ValueError('no valid MLP dims are given')

    def forward(self, graph, x, twp=False, tasks=None):
        graph = graph.local_var().to('cuda:{}'.format(self.gpu))
        e_list = []
        x = self.neighbor_agg(graph, x)
        logits, e = self.feat_trans(graph, x, twp=twp, cls=tasks)
        e_list = e_list + e
        return logits, e_list

    def feat_trans(self, graph, x, twp=False, cls=None):
        for i, layer in enumerate(self.feat_trans_layers[:-1]):
            x = layer(x)
            if self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.feat_trans_layers[-1](x)

        self.second_last_h = x

        mask = torch.zeros(x.shape[-1], device=x.get_device())
        if cls is not None:
            mask[cls] = 1.
        else:
            mask[:] = 1.
        x = x * mask
        # for twp
        elist = []
        if self.twp:
            graph.srcdata['h'] = x
            graph.apply_edges(
                lambda edges: {'e': torch.sum((torch.mul(edges.src['h'], torch.tanh(edges.dst['h']))), 1)})
            e = self.leaky_relu(graph.edata.pop('e'))
            e_soft = edge_softmax(graph, e)

            elist.append(e_soft)

        return x, elist
        #return x.log_softmax(dim=-1), elist
    def reset_params(self):
        for layer in self.feat_trans_layers:
            layer.reset_parameters()

class CustomFDGNN_Agg(nn.Module):
    # only the neighborhood aggregation of CustomFDGNN
    def __init__(self, dim, k=1, cached=False, norm=None, allow_zero_in_degree=False):
        super().__init__()
        #self.dims = dims
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.W_center = []
        self.W_nb = []
        for l in range(1):
            self.W_center.append(init.kaiming_uniform_(torch.Tensor(dim, dim)))
            self.W_nb.append(init.kaiming_uniform_(torch.Tensor(dim, dim)))

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute Simplifying Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

        Note
        ----
        If ``cache`` is set to True, ``feat`` and ``graph`` should not change during
        training, or you will get wrong results.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X
                for _ in range(self._k):
                    feat = feat * norm
                    graph.ndata['h'] = feat.mm(self.W_nb[_].cuda(feat.get_device()))
                    graph.ndata['center'] = feat.mm(self.W_center[_].cuda(feat.get_device()))
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'center'))
                    feat = graph.ndata.pop('center')
                    feat = feat * norm

                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat
            # return self.fc(feat)
            return feat


class CustomFDGNN(nn.Module):
    # a reservoir computing module mentioned by the reviewer
    def __init__(self, args):
        super().__init__()
        linear_layer = linear_choices[args.CustomFDGNN_args['linear']]
        if args.method == 'twp':
            self.twp=True
        else:
            self.twp=False
        self.bn = args.CustomFDGNN_args['batch_norm']
        self.dropout = args.CustomFDGNN_args['dropout']
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.gpu = args.gpu
        self.neighbor_agg = CustomFDGNN_Agg(args.d_data,k=args.CustomFDGNN_args['k'])
        self.feat_trans_layers = nn.ModuleList()
        if self.bn:
            self.bns = nn.ModuleList()
        h_dims = args.CustomFDGNN_args['h_dims']
        if len(h_dims) > 0:
            self.feat_trans_layers.append(linear_layer(args.d_data, h_dims[0], bias=args.CustomFDGNN_args['linear_bias']))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(h_dims[0]))
            for i in range(len(h_dims) - 1):
                self.feat_trans_layers.append(linear_layer(h_dims[i], h_dims[i + 1], bias=args.CustomFDGNN_args['linear_bias']))
                if self.bn:
                    self.bns.append(nn.BatchNorm1d(h_dims[i + 1]))
            self.feat_trans_layers.append(linear_layer(h_dims[-1], args.n_cls, bias=args.CustomFDGNN_args['linear_bias']))
        elif len(h_dims) == 0:
            self.feat_trans_layers.append(linear_layer(args.d_data, args.n_cls, bias=args.CustomFDGNN_args['linear_bias']))
        else:
            raise ValueError('no valid MLP dims are given')

    def forward(self, graph, x, twp=False, tasks=None):
        graph = graph.local_var().to('cuda:{}'.format(self.gpu))
        e_list = []
        x = self.neighbor_agg(graph, x)
        logits, e = self.feat_trans(graph, x, twp=twp, cls=tasks)
        e_list = e_list + e
        return logits, e_list

    def feat_trans(self, graph, x, twp=False, cls=None):
        for i, layer in enumerate(self.feat_trans_layers[:-1]):
            x = layer(x)
            if self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.feat_trans_layers[-1](x)

        self.second_last_h = x

        mask = torch.zeros(x.shape[-1], device=x.get_device())
        if cls is not None:
            mask[cls] = 1.
        else:
            mask[:] = 1.
        x = x * mask
        # for twp
        elist = []
        if self.twp:
            graph.srcdata['h'] = x
            graph.apply_edges(
                lambda edges: {'e': torch.sum((torch.mul(edges.src['h'], torch.tanh(edges.dst['h']))), 1)})
            e = self.leaky_relu(graph.edata.pop('e'))
            e_soft = edge_softmax(graph, e)

            elist.append(e_soft)

        return x, elist
        #return x.log_softmax(dim=-1), elist
    def reset_params(self):
        for layer in self.feat_trans_layers:
            layer.reset_parameters()

