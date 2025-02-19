import torch.nn.functional as F
from .gnns import GAT, GCN, GIN, CustomDecoupledSGC, CustomDecoupledS2GC, CustomDecoupledAPPNP, CustomFDGNN

def get_model(dataset, args):
    n_classes = args.n_cls_per_task
    print('n_classes', n_classes)
    if args.backbone == 'GAT':
        heads = ([args.GAT_args['heads']] * args.GAT_args['num_layers']) + [args.GAT_args['out_heads']]
        model = GAT(args, heads, F.elu)
    elif args.backbone == 'GCN':
        model = GCN(args)
    elif args.backbone == 'GIN':
        model = GIN(args)
    elif args.backbone in ['CustomDecoupledSGC', 'CustomDecoupledS2GC', 'CustomDecoupledAPPNP', 'CustomFDGNN']:
        PDGNN = {'CustomDecoupledSGC':CustomDecoupledSGC, 'CustomDecoupledS2GC':CustomDecoupledS2GC, 'CustomDecoupledAPPNP':CustomDecoupledAPPNP, 'CustomFDGNN':CustomFDGNN}
        model = PDGNN[args.backbone](args)
    return model
