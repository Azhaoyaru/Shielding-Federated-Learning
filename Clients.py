import time

import matplotlib.pyplot as plt
from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.utils import generate_data_loaders_from_distributed_dataset
from federated_learning.datasets.data_distribution import distribute_batches_equally
from federated_learning.utils import average_nn_parameters
from federated_learning.utils import convert_distributed_data_into_numpy
from federated_learning.utils import poison_data
from federated_learning.utils import identify_random_elements
from federated_learning.utils import save_results
from federated_learning.utils import load_train_data_loader
from federated_learning.utils import load_test_data_loader
from federated_learning.utils import generate_experiment_ids
from federated_learning.utils import convert_results_to_csv
from federated_learning.dimensionality_reduction import calculate_pca_of_gradients
from client import Client
from sklearn.cluster import KMeans
import argparse
import random, os
from util import *
import torch.cuda
import torchvision.transforms as transforms
#import tensorflow
from torch import Tensor
from torch import nn
import numpy as np
from numpy import *

class Logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "a+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

parser = argparse.ArgumentParser()
parser.add_argument('--poison-label', '-plabel', default=8, type=int, help='label of the poisons, or the target label we want to classify into')
parser.add_argument('--poison-opt', default='adam', type=str)
parser.add_argument('--poison-lr', '-plr', default=4e-2, type=float, help='learning rate for making poison')
parser.add_argument('--poison-momentum', '-pm', default=0.9, type=float, help='momentum for making poison')
parser.add_argument('--poison-decay-ites', type=int, metavar='int', nargs="+", default=[])
parser.add_argument('--poison-decay-ratio', default=0.1, type=float)
parser.add_argument('--poison-epsilon', '-peps', default=0.1, type=float, help='maximum deviation for each pixel')

parser.add_argument('--poison-ites', default=500, type=int, help='iterations for making poison')  #MobileNetV2:500

parser.add_argument('--tol', default=1e-6, type=float)
parser.add_argument('--net-repeat', default=1, type=int)
parser.add_argument('--chk-path', default='chk-black', type=str, help="save results .pth files of the BPoLT methods")
parser.add_argument('--resume-poison-ite', default=0, type=int,help="Will automatically match the poison checkpoint corresponding to this iteration and resume training")

parser.add_argument('--subs-dp', default=[0.3], nargs="+", type=float,help='Dropout for the surrogate nets, will be turned on for both training and testing')
parser.add_argument("--subs-pretrained-name", default=['ckpt_subs-dp%.3f-%s-seed%d.t7'], nargs="+", type=str)

argse = parser.parse_args()  #为了区分上边的args，这里设置argse
argse.pid = os.getpid() #返回一个整数值，表示当前进程的进程ID

def train_subset_of_clients(epoch, args, clients, poisoned_attackers):
    """
    Train a subset of clients per round.
    """
    kwargs = args.get_round_worker_selection_strategy_kwargs()
    kwargs["current_epoch_number"] = epoch
    Attack_sucess_rate_all = []

    random_workers = args.get_round_worker_selection_strategy().select_round_workers(list(range(args.get_num_workers())), poisoned_attackers, kwargs)

    for client_idx in random_workers:
        args.get_logger().info("Training epoch #{} on client #{}", str(epoch), str(clients[client_idx].get_client_index()))

        if client_idx in poisoned_attackers:
            Attack_sucess_rate=clients[client_idx].attack_sucess_rate(epoch,client_idx)
            Attack_sucess_rate = list(Attack_sucess_rate)
            Attack_sucess_rate = Attack_sucess_rate[0]
            Attack_sucess_rate_all.append(Attack_sucess_rate)
        else:
            clients[client_idx].train(epoch)
    final_attacker_sucess_rate = sum(Attack_sucess_rate_all) / (len(poisoned_attackers))

    parameters = [clients[client_idx].get_nn_parameters() for client_idx in random_workers]
    new_nn_params = average_nn_parameters(parameters)

    for client in clients:
        args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
        client.update_nn_parameters(new_nn_params)

    return clients[0].test(epoch), random_workers

def create_clients(args, train_data_loaders, test_data_loader):
    """
    Create a set of clients.
    """
    clients = []
    for idx in range(args.get_num_workers()):

        clients.append(Client(args, idx, train_data_loaders[idx], test_data_loader))

    return clients

def run_machine_learning(clients, args, poisoned_attackers):
    """
    Complete machine learning over a series of clients.
    """
    epoch_test_set_results = []
    worker_selection = []
    #Attack_sucess_rate_all=[]

    for epoch in range(1, args.get_num_epochs() + 1):

        results, workers_selected = train_subset_of_clients(epoch, args, clients, poisoned_attackers)

        epoch_test_set_results.append(results)
        worker_selection.append(workers_selected)

    return convert_results_to_csv(epoch_test_set_results)#, worker_selection

#根据攻击者情况调整
def run_exp(replacement_method, num_poison_attackers, KWARGS, client_selection_strategy, idx):

    log_files, results_files, models_folders, worker_selections_files = generate_experiment_ids(idx, 1)
    handler = logger.add(log_files[0], enqueue=True)  # Initialize logger

    args = Arguments(logger)
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_poison_attackers)
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.log()

    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)

    distributed_train_dataset = distribute_batches_equally(train_data_loader, args.get_num_workers()) #50
    distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset) #50

    #perform BPoLT
     # 获得噪声数据，更新最后生成的训练数据集

    train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size())
    clients = create_clients(args, train_data_loaders, test_data_loader)

    victim_client = clients[random.randint(0, len(clients))]
    victim_net=victim_client.net.cuda()

    sub_net_list = []
    for n_chk, chk_name in enumerate(argse.subs_pretrained_name):
        for snet in argse.substitute_nets:
            if argse.subs_dp[n_chk] > 0.0:
                sub_net = load_pretrained_net(snet, chk_name, model_chk_path=argse.model_resume_path,
                                              test_dp=argse.subs_dp[n_chk], seed=argse.seed)
            elif argse.subs_dp[n_chk] == 0.0:
                sub_net = load_pretrained_net(snet, chk_name, model_chk_path=argse.model_resume_path)
            else:
                assert False
            sub_net_list.append(sub_net)

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)
    transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize(cifar10_mean, cifar10_std),])

    target = fetch_target(argse.target_label, argse.target_index, 50, subset='others', path=argse.train_data_path, transforms=transform_test)

    base_poison_images, base_poison_images_idxs = fetch_poison_bases(argse.poison_label, argse.poison_num, subset='others',
                                                         path=argse.train_data_path, transforms=transform_test)
    chk_path = os.path.join(argse.chk_path, argse.mode)
    chk_path=os.path.join(chk_path,str(argse.poison_ites))
    chk_path = os.path.join(chk_path, str(argse.target_index))
    if not os.path.exists(chk_path):
        os.makedirs(chk_path)
    import sys
    # 创建log.txt文件
    sys.stdout = Logger('{}/log.txt'.format(chk_path))

    if argse.resume_poison_ite>0:
        state_dict = torch.load(os.path.join(chk_path, "poison_%05d.pth" % argse.resume_poison_ite))

        poison_tuple_list, base_poison_images_idxs = state_dict['poison'], state_dict['idx']
        poison_init = [pt.to('cuda') for pt, _ in poison_tuple_list]
        # re-direct the results to the resumed dir...
        chk_path += '-resume'
        if not os.path.exists(chk_path):
            os.makedirs(chk_path)
    else:
        poison_images_init = base_poison_images

    print("Selected base image indices: {}".format(base_poison_images_idxs))
    #poison_images_init = base_poison_images

    t2_genpoison0 = time.time()  # t1 = time.time()
    poison_tuple_list = make_Label_Transfer_poisons(sub_net_list, victim_net, base_poison_images, target, device='cuda',
                  opt_method=argse.poison_opt, lr=argse.poison_lr, momentum=argse.poison_momentum, iterations=argse.poison_ites,
                  epsilon=argse.poison_epsilon, decay_ites=argse.poison_decay_ites, decay_ratio=argse.poison_decay_ratio,
                  mean=torch.Tensor(cifar10_mean).reshape(1, 3, 1, 1), std=torch.Tensor(cifar10_std).reshape(1, 3, 1, 1),
                  chk_path=chk_path, poison_idxes=base_poison_images_idxs, poison_label=argse.poison_label, tol=argse.tol,
                  end2end=argse.end2end, start_ite=argse.resume_poison_ite, poison_init=poison_images_init,
                  mode=argse.mode, net_repeat=argse.net_repeat)
    #tt1 = time.time()

    #PCA降维后的目标样本和投毒样本
    #生成初始投毒区域
    #计算重心更新投毒区域
    #获得新的training dataset by BPoLT
    distributed_train_dataset

    train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size())

    # 客户端训练开始
    clients = create_clients(args, train_data_loaders, test_data_loader)

    results, worker_selection = run_machine_learning(clients, args, poisoned_attackers) #原
    #results = run_machine_learning(clients, args, poisoned_attackers)

    #保存的.csv文件
    save_results(results, results_files[0])
    save_results(worker_selection, worker_selections_files[0])

    logger.remove(handler)
