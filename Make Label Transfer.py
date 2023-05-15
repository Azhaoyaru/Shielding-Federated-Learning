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

from federated_learning.utils import generate_experiment_ids
from federated_learning.utils import convert_results_to_csv
from client import Client

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

parser.add_argument('--mode', default='BPoLT', type=str)
parser.add_argument('--poison-num', default=3, type=int, help='number of poisons') #5
parser.add_argument('--train-data-path', default='data/CIFAR10_TRAIN_Split.pth', type=str, help='path to the official datasets')
parser.add_argument('--model-resume-path', default='pretrained_models', type=str, help="Path to the pre-trained models")

parser.add_argument('--subs-dp', default=[0.3], nargs="+", type=float,help='Dropout for the surrogate nets, will be turned on for both training and testing')
parser.add_argument("--subs-pretrained-name", default=['ckpt_subs-dp%.3f-%s-seed%d.t7'], nargs="+", type=str)

#选择 预训练网络模型
# parser.add_argument('--surrogate-nets', default=['ResNet50'], nargs="+", required=False)
parser.add_argument('--surrogate-nets', default=['ResNet18'], nargs="+", required=False)
# parser.add_argument('--surrogate-nets', default=['MobileNetV2'], nargs="+", required=False)
# parser.add_argument('--surrogate-nets', default=['SENet18'], nargs="+", required=False)
# parser.add_argument('--surrogate-nets', default=['DPN26'], nargs="+", required=False)

parser.add_argument('--net', type=str, default='ResNet18')  #net=['SENet18','ResNet50', 'MobileNetV2','DPN92','MobileNetV2']
parser.add_argument('--seed', type=int, default=1226) #设定随机数的种子，目的是为了让结果具有重复性，重现结果。

argse = parser.parse_args()  #为了区分上边的args，这里设置argse
argse.pid = os.getpid() #返回一个整数值，表示当前进程的进程ID

def make_Label_Transfer_poisons(subs_net_list, victim_net, base_poison_image_list, target, device, opt_method='adam',
                             lr=0.1,momentum=0.9, iterations=10, epsilon=0.1,decay_ites=[10000, 15000], decay_ratio=0.1, #iterations=4000
                             mean=torch.Tensor((0.4914, 0.4822, 0.4465)).reshape(1, 3, 1, 1),
                             std=torch.Tensor((0.2023, 0.1994, 0.2010)).reshape(1, 3, 1, 1),
                             chk_path='', poison_idxes=[], poison_label=-1,
                             tol=1e-6, start_ite=0, poison_init=None, end2end=False, mode='convex', net_repeat=1):
    victim_net.eval()

    poison_batch = PoisonBatch(poison_init).to(device)

    opt_method = opt_method.lower()
    if opt_method == 'sgd':
        optimizer = torch.optim.SGD(poison_batch.parameters(), lr=lr, momentum=momentum)
    elif opt_method == 'adam':
        optimizer = torch.optim.Adam(poison_batch.parameters(), lr=lr, betas=(momentum, 0.999))
    target = target.to(device)
    std, mean = std.to(device), mean.to(device)
    base_tensor_batch = torch.stack(base_poison_image_list, 0).to(device) #base_poison_image_list:3：(3,32,32)
    base_range01_batch = (base_tensor_batch * std + mean).to(device) #(3,3,32,32) 归一化处理：为了后续扰动,控制基本投毒图像批次数据在0-1之间

    # Because we have turned on DP for the surrogate networks,the target image's feature becomes random.
    # We can try enforcing the convex polytope in one of the multiple realizations of the feature,but empirically one realization is enough.
    target_feat_list = []
    #凸组合的系数 Coefficients for the convex combination.
    #从上一步的系数初始化可以加快收敛速度。Initializing from the coefficients of last step gives faster convergence.
    sub_init_coeff_list = []
    n_poisons = len(base_poison_image_list)  #投毒数量的个数poison_num=3
    for n, subnet in enumerate(subs_net_list):
        subnet.eval()

        target_feat_list.append(subnet(x=target, penu=True).detach()) #返回目标特征集--mnist dataset
        sub_coeff = torch.ones(n_poisons, 1).to(device) / n_poisons  #返回张量（5,1）替代网络的coeff

        sub_init_coeff_list.append(sub_coeff)

    # Keep this for evaluation.
    target_feat_in_target = victim_net(x=target, penu=True).detach() #mnist dataset
    target_init_coeff = [torch.ones(len(base_poison_image_list), 1).to(device) / n_poisons]

    bpolt_loss_func = get_BPoLT_loss_end2end

    coeffs_time = 0

    for ite in range(start_ite, iterations): #开始迭代---
        if ite in decay_ites: #false
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_ratio
            print("%s Iteration %d, Adjusted lr to %.2e" % (time.strftime("%Y-%m-%d %H:%M:%S"), ite, lr))

        poison_batch.zero_grad()
        #t = time.time()
        if mode == 'convex': #True
            total_loss, sub_init_coeff_list, coeffs_time_tmp = bpolt_loss_func(subs_net_list, target_feat_list, poison_batch,
                                                           sub_init_coeff_list, net_repeat=net_repeat, tol=tol)
        elif mode == 'mean':
            total_loss = loss_from_center(subs_net_list, target_feat_list, poison_batch, net_repeat, end2end)
            coeffs_time_tmp = 0

        elif mode.startswith('coeffs_fixed_type_'):
            coeffs_type = int(mode.split("coeffs_fixed_type_")[1])
            coeffs = COEFFS[coeffs_type]
            random.shuffle(coeffs)
            coeffs = torch.Tensor([coeffs]).t().to(device)
            assert abs(sum(coeffs).item() - 1.0) < 10e-3, print(sum(coeffs).item())
            if ite == start_ite:
                print("coeffs fixed to: {}".format(coeffs))
            total_loss = loss_when_coeffs_fixed(subs_net_list, target_feat_list, poison_batch, coeffs, net_repeat, end2end)
            coeffs_time_tmp = 0

        coeffs_time += coeffs_time_tmp
        #print("计算损失耗时：", coeffs_time)
        total_loss.backward()
        optimizer.step()

    #clip the perturbations into the range
        assert epsilon == 0.1
        #torch.clamp()
        perturb_range01 = torch.clamp((poison_batch.poison.data - base_tensor_batch) * std, -epsilon, epsilon)
        perturbed_range01 = torch.clamp(base_range01_batch.data + perturb_range01.data, 0, 1)
        poison_batch.poison.data = (perturbed_range01 - mean) / std

        if ite % 50 == 0 or ite == iterations - 1: #iterations=4000
            target_loss, target_init_coeff, _ = bpolt_loss_func([victim_net], [target_feat_in_target],
                                                             poison_batch, target_init_coeff, net_repeat=1, tol=tol)

            # compute the difference in target
            print(" %s Iteration %d \t Training Loss: %.3e \t Loss in Target Net: %.3e\t  " % (
                time.strftime("%Y-%m-%d %H:%M:%S"), ite, total_loss.item(), target_loss.item()))
            print("Update_coeffs_time for BPoLT of Algorithm_1:", coeffs_time) #add
            sys.stdout.flush()

            #对基本投毒图像增加扰动并实施投毒来生成的投毒数据  save the checkpoints
            poison_tuple_list = get_poison_tuples(poison_batch, poison_label)

            #保存为.pth文件
            torch.save({'poison': poison_tuple_list, 'idx': poison_idxes, 'Update_coeffs_time for BPoLT of Algorithm_1': coeffs_time,
                        'target_loss': target_loss, 'Training_loss': total_loss,
                        'coeff_list': sub_init_coeff_list, 'coeff_list_in_victim': target_init_coeff},
                         os.path.join(chk_path, "poison_%05d.pth" % ite))

    return get_poison_tuples(poison_batch, poison_label)
