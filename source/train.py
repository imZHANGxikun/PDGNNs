import traceback
import argparse
from training.utils import set_seed, remove_illegal_characters,str2dict,compose_hyper_params,assign_hyp_param
from distutils.util import strtobool
#from visualize import *
from pipeline import *
import sys

dir_home = os.getcwd()
sys.path.append(os.path.join(dir_home,'.local/lib/python3.7/site-packages')) # for hpc usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TEM')
    parser.add_argument("--dataset", type=str, default='CoraFull-CL', help='Products-CL, Reddit-CL, Arxiv-CL, CoraFull-CL')
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to use.")
    parser.add_argument("--seed", type=int, default=1, help="seed for exp")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs, default = 200")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4, help="weight decay")
    parser.add_argument('--backbone', type=str, default='CustomDecoupledAPPNP', help="backbone GNN, [GAT, GCN, GIN]")
    parser.add_argument('--method', type=str,
                        choices=["bare", 'lwf', 'gem', 'ewc', 'mas', 'twp', 'jointtrain', 'ergnn', 'joint','Joint','subgraph_replay','TEM'], default="TEM",
                        help="baseline continual learning method")
    # parameters for continual learning settings
    parser.add_argument('--share-labels', type=strtobool, default=False,
                        help='task-IL specific, whether to share output label space for different tasks')
    parser.add_argument('--inter-task-edges', type=strtobool, default=False,
                        help='whether to keep the edges connecting nodes from different tasks')
    parser.add_argument('--classifier-increase', type=strtobool, default=True,
                        help='(deprecated) class-IL specific, whether to enlarge the label space with the coming of new classes, unrealistic to be set as False')

    # extra parameters
    parser.add_argument('--refresh_data', type=strtobool, default=False, help='whether to load existing splitting or regenerate')
    parser.add_argument('--d_dtat', default=None, help='will be assigned during running')
    parser.add_argument('--n_cls', default=None, help='will be assigned during running')
    parser.add_argument('--ratio_valid_test', nargs='+', default=[0.2, 0.2], help='ratio of nodes used for valid and test')
    parser.add_argument('--transductive', type=strtobool, default=True, help='using transductive or inductive')
    parser.add_argument('--default_split', type=strtobool, default=False, help='whether to  use the data split provided by the dataset')
    parser.add_argument('--task_seq', default=[])
    parser.add_argument('--n-task', default=0, help='will be assigned during running')
    parser.add_argument('--n_cls_per_task', default=2, help='how many classes does each task  contain')
    parser.add_argument('--CustomDecoupledS2GC_args',
                        default={'h_dims': [256], 'dropout': 0.0, 'bias': False, 'k': 1, 'alpha': 0.05,
                                 'batch_norm': False, 'linear_bias': False, 'linear': 'nn.Linear'})
    parser.add_argument('--CustomDecoupledAPPNP_args',
                        default={'h_dims': [256], 'dropout': 0.0, 'bias': False, 'k': 2, 'alpha': 0.05,
                                 'batch_norm': False, 'linear_bias': False, 'linear': 'nn.Linear'})
    parser.add_argument('--CustomDecoupledSGC_args',
                        default={'h_dims': [256], 'dropout': 0.0, 'bias': False, 'k': 1, 'alpha': 0.05,
                                 'batch_norm': False, 'linear_bias': False, 'linear': 'nn.Linear'})
    parser.add_argument('--GAT-args',
                        default={'num_layers': 1, 'num_hidden': 32, 'heads': 8, 'out_heads': 1, 'feat_drop': .6,
                                 'attn_drop': .6, 'negative_slope': 0.2, 'residual': False})
    parser.add_argument('--GCN-args', default={'h_dims': [256], 'dropout': 0.0, 'batch_norm': False})
    parser.add_argument('--GIN-args', default={'h_dims': [256], 'dropout': 0.0})
    parser.add_argument('--SGC_args', default={'h_dims': [256], 'dropout': 0.0, 'bias': False, 'k': 2, 'alpha': 0.05, 'batch_norm': False, 'linear_bias': False, 'linear': 'nn.Linear'})
    parser.add_argument('--ergnn_args', type=str2dict, default={'budget': [100,1000], 'd': [0.5], 'sampler': ['CM']},
                        help='sampler options: CM, CM_plus, MF, MF_plus')
    parser.add_argument('--tem_args', default={'budget': 400, 'sampler': 'cover_max_select_02'})
    parser.add_argument('--lwf_args', type=str2dict, default={'lambda_dist': [1.0, 10.0], 'T': [2.0, 20.0]})
    parser.add_argument('--twp_args', type=str2dict, default={'lambda_l': 10000., 'lambda_t': 10000., 'beta': 0.01})
    parser.add_argument('--ewc_args', type=str2dict, default={'memory_strength': 1000000.})
    parser.add_argument('--mas_args', type=str2dict, default={'memory_strength': 10000.})
    parser.add_argument('--gem_args', type=str2dict, default={'memory_strength': 0.5, 'n_memories': 100})
    parser.add_argument('--bare_args', type=str2dict, default={'Na': None})
    parser.add_argument('--joint_args', type=str2dict, default={'Na': None})
    parser.add_argument('--sgreplay_args', type=str2dict,
                        default={'sampler': 'random_walk', 'c_node_budget': 10, 'nei_budget': [1, 1], 'lambda': 1})
    parser.add_argument('--cls-balance', type=strtobool, default=True, help='whether to balance the cls when training and testing')
    parser.add_argument('--repeats', type=int, default=5, help='how many times to repeat the experiments for the mean and std')
    parser.add_argument('--ILmode', default='classIL',choices=['taskIL','classIL'])
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--minibatch', type=strtobool, default=False, help='whether to use the mini-batch training')
    parser.add_argument('--batch_shuffle', type=strtobool, default=False, help='whether to shuffle the data when constructing the dataloader')
    parser.add_argument('--sample_nbs', type=strtobool, default=True, help='whether to sample neighbors instead of using all')
    parser.add_argument('--n_nbs_sample', type=lambda x: [int(i) for i in x.replace(' ', '').split(',')], default=[10, 20], help='number of neighbors to sample per hop, use comma to separate the numbers when using the command line, e.g. 10,25 or 10, 25')
    parser.add_argument('--nb_sampler', default=None)
    parser.add_argument('--replace_illegal_char', type=strtobool, default=False)
    parser.add_argument('--ori_data_path', type=str, default='/store/data', help='the root path to raw data')
    parser.add_argument('--data_path', type=str, default='./data', help='the path to processed data (splitted into tasks)')
    parser.add_argument('--result_path', type=str, default='./results', help='the path for saving results')
    parser.add_argument('--overwrite_result', type=strtobool, default=False, help='whether to overwrite existing results')
    parser.add_argument('--load_check', type=strtobool, default=False, help='whether to check the existence of processed data by loading')
    parser.add_argument('--perform_testing', type=strtobool, default=True, help='whether to check the existence of processed data by loading')
    args = parser.parse_args()
    args.ratio_valid_test = [float(i) for i in args.ratio_valid_test]
    set_seed(args)
    mkdir_if_missing(f'{args.data_path}')

    method_args = {'TEM': args.tem_args, 'ergnn': args.ergnn_args, 'lwf': args.lwf_args, 'twp': args.twp_args, 'ewc': args.ewc_args,
                   'bare': args.bare_args, 'gem': args.gem_args, 'mas': args.mas_args, 'joint': args.joint_args, 'subgraph_replay': args.sgreplay_args}
    backbone_args = {'CustomDecoupledSGC': args.CustomDecoupledSGC_args,
                     'CustomDecoupledS2GC': args.CustomDecoupledSGC_args,
                     'CustomDecoupledAPPNP': args.CustomDecoupledAPPNP_args, 'GCN': args.GCN_args, 'GAT': args.GAT_args, 'GIN': args.GIN_args, 'SGC': args.SGC_args}
    hyp_param_list = compose_hyper_params(method_args[args.method])
    AP_best, name_best = 0, None
    AP_dict = {str(hyp_params).replace("'",'').replace(' ','').replace(',','_').replace(':','_'):[] for hyp_params in hyp_param_list}
    AF_dict = {str(hyp_params).replace("'",'').replace(' ','').replace(',','_').replace(':','_'):[] for hyp_params in hyp_param_list}
    PM_dict = {str(hyp_params).replace("'",'').replace(' ','').replace(',','_').replace(':','_'):[] for hyp_params in hyp_param_list}
    for hyp_params in hyp_param_list:
        # iterate over each candidate hyper-parameter combination, and find the best one over the validation set
        hyp_params_str = str(hyp_params).replace("'",'').replace(' ','').replace(',','_').replace(':','_')
        print(hyp_params_str)
        assign_hyp_param(args,hyp_params)
        args.nb_sampler = dgl.dataloading.MultiLayerNeighborSampler(args.n_nbs_sample) if args.sample_nbs else dgl.dataloading.MultiLayerFullNeighborSampler(2)
        main = get_pipeline(args)
        train_ratio = round(1 - args.ratio_valid_test[0] - args.ratio_valid_test[1], 2)
        if args.ILmode == 'classIL':
            subfolder = f'inter_task_edges/cls_IL/train_ratio_{train_ratio}/' if args.inter_task_edges else f'no_inter_task_edges/cls_IL/train_ratio_{train_ratio}/'
        elif args.ILmode == 'taskIL':
            subfolder = f'inter_task_edges/tsk_IL/train_ratio_{train_ratio}/' if args.inter_task_edges else f'no_inter_task_edges/tsk_IL/train_ratio_{train_ratio}/'

        name = f'{subfolder}val_{args.dataset}_{args.n_cls_per_task}_{args.method}_{list(hyp_params.values())}_{args.backbone}_{backbone_args[args.backbone]}_{args.classifier_increase}_{args.cls_balance}_{args.epochs}_{args.repeats}'
        if args.minibatch:
            name = name + f'_bs{args.batch_size}'
        mkdir_if_missing(f'{args.result_path}/' + subfolder)
        if args.replace_illegal_char:
            name = remove_illegal_characters(name)

        if os.path.isfile(f'{args.result_path}/{name}.pkl') and not args.overwrite_result:
            # if the results exist, directly use them without over-writting
            print('The results of the following configuration exists, will load this result for use \n', f'{args.result_path}/{name}.pkl')
            AP_AF = show_final_APAF(f'{args.result_path}/{name}.pkl')
            AP = float(AP_AF.split('$')[0])
            if AP > AP_best:
                AP_best = AP
                hyp_best_str = hyp_params_str
                name_best = name
                print(f'best params is {hyp_best_str}, best AP is {AP_best}')
            continue
        else:
            # if results do not exist or choose to overwrite existing results
            acc_matrices = []
            print('method args are', hyp_params)
            for ite in range(args.repeats):
                print(name, ite)
                args.current_model_save_path = [name,ite]
                try:
                    AP, AF, acc_matrix = main(args,valid=True)
                    AP_dict[hyp_params_str].append(AP)

                    # choose the best configuration according to the validation results
                    acc_matrices.append(acc_matrix)
                    torch.cuda.empty_cache()
                    if ite == 0:
                        with open(
                                f'{args.result_path}/log.txt',
                                'a') as f:
                            f.write(name)
                            f.write('\nAP:{},AF:{}\n'.format(AP, AF))
                except Exception as e:
                    mkdir_if_missing(f'{args.result_path}/errors/' + subfolder)
                    if ite > 0:
                        name_ = f'{subfolder}val_{args.dataset}_{args.n_cls_per_task}_{args.method}_{list(hyp_params.values())}_{args.backbone}_{backbone_args[args.backbone]}_{args.classifier_increase}_{args.cls_balance}_{args.epochs}_{ite}'
                        with open(f'{args.result_path}/{name_}.pkl', 'wb') as f:
                            pickle.dump(acc_matrices, f)
                    print('error', e)
                    name = 'errors/{}'.format(name)
                    acc_matrices = traceback.format_exc()
                    print(acc_matrices)
                    print('error happens on \n', name)
                    torch.cuda.empty_cache()
                    break
            if np.mean(AP_dict[hyp_params_str]) > AP_best:
                AP_best = np.mean(AP_dict[hyp_params_str])
                hyp_best_str = hyp_params_str
                name_best = name
            print(f'best params is {hyp_best_str}, best AP is {AP_best}')
            with open(f'{args.result_path}/{name}.pkl', 'wb') as f:
                pickle.dump(acc_matrices, f)

    # save the models
    config_name = name_best.split('/')[-1]
    subfolder_c = name_best.split(config_name)[-2]
    if args.perform_testing:
        print('----------Now in testing--------')
        acc_matrices = []
        for ite in range(args.repeats):
            args.current_model_save_path = [name_best, ite]
            AP_test, AF_test, acc_matrix_test = main(args, valid=False)
            acc_matrices.append(acc_matrix_test)
        with open(f'{args.result_path}/{name_best}.pkl'.replace('val', 'te'), 'wb') as f:
            pickle.dump(acc_matrices, f)

