import numpy as np, sys, os, random, pdb, json, uuid, time, argparse
from pprint import pprint
import logging, logging.config
from collections import defaultdict as ddict
from ordered_set import OrderedSet

# PyTorch related imports
import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from torch.utils.data import DataLoader
from torch.nn import Parameter
from torch_scatter import scatter_add
import math
#np.set_printoptions(precision=4)

import random

torch.cuda.manual_seed_all(47)
torch.manual_seed(48)
np.random.seed(49)
random.seed(50)

#torch1.8没有irfft属性，为了兼容采取下面方法
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft2
    from torch.fft import rfft2
    def rfft(x, d):
        t = rfft2(x, dim = (-d))
        return torch.stack((t.real, t.imag), -1)
    def irfft(x, d, signal_sizes):
        return irfft2(torch.complex(x[:,:,0], x[:,:,1]), s = signal_sizes, dim = (-d))


def set_gpu(gpus):
    """
    Sets the GPU to be used for the run

    Parameters
    ----------
    gpus:           List of GPUs to be used for the run

    Returns
    -------

    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def get_logger(name, log_dir, config_dir):
    """
    Creates a logger object

    Parameters
    ----------
    name:           Name of the logger file
    log_dir:        Directory where logger file needs to be stored
    config_dir:     Directory from where log_config.json needs to be read

    Returns
    -------
    A logger object which writes to both file and stdout

    """
    config_dict = json.load(open(os.path.join(config_dir, 'log_config.json')))
    config_dict['handlers']['file_handler']['filename'] = os.path.join(log_dir, name.replace('/', '-')) #设置日志文件名
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s' #%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s

    #std_out_format = '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s' #这里的配置没啥用，仅仅受json配置的影响

    #std_out_format = '%(asctime)s  - [%(levelname)s] - %(message)s' ##这里的配置没啥用，仅仅受json配置的影响
    #这里的格式匹设置的是屏幕输出的格式，而不是写入文件的格式，文件的格式在json里面
    
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


def get_combined_results(left_results, right_results):
    results = {}
    count = float(left_results['count'])

    results['left_mr'] = round(left_results['mr'] / count, 5)
    results['left_mrr'] = round(left_results['mrr'] / count, 5)
    results['right_mr'] = round(right_results['mr'] / count, 5)
    results['right_mrr'] = round(right_results['mrr'] / count, 5)
    results['mr'] = round((left_results['mr'] + right_results['mr']) / (2 * count), 5)
    results['mrr'] = round((left_results['mrr'] + right_results['mrr']) / (2 * count), 5)

    for k in range(10):
        results['left_hits@{}'.format(k + 1)] = round(left_results['hits@{}'.format(k + 1)] / count, 5)
        results['right_hits@{}'.format(k + 1)] = round(right_results['hits@{}'.format(k + 1)] / count, 5)
        results['hits@{}'.format(k + 1)] = round(
            (left_results['hits@{}'.format(k + 1)] + right_results['hits@{}'.format(k + 1)]) / (2 * count), 5)
    return results


def analy_get_combined_results(left_results, right_results):
    results = {}
    count = 2

    results['count'] = left_results['count']
    results['left_mr'] = left_results['mr'] / count
    results['left_mrr'] = left_results['mrr'] / count
    results['right_mr'] = right_results['mr'] / count
    results['right_mrr'] = right_results['mrr'] / count
    results['mr']  = (left_results['mr'] + right_results['mr']) / (count)
    results['mrr'] = (left_results['mrr'] + right_results['mrr']) / (count)

    for k in range(10):
        results['left_hits@{}'.format(k + 1)] = left_results['hits@{}'.format(k + 1)] / count
        results['right_hits@{}'.format(k + 1)] = right_results['hits@{}'.format(k + 1)] / count
        results['hits@{}'.format(k + 1)] = (left_results['hits@{}'.format(k + 1)] + right_results['hits@{}'.format(k + 1)]) / (count)
        
    return results


def get_param(shape):
    param = Parameter(torch.Tensor(*shape))
    xavier_normal_(param.data)
    return param



def get_xavier_uniform(shape):
	param = Parameter(torch.Tensor(*shape)); #注意这是带参数的，是要梯度更新的	
	xavier_uniform_(param.data)
	return param

def get_uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

def sp_get_uniform(size, tensor):
    if tensor is not None:
        tensor.data.uniform_()
        
def get_normal(tensor):
    if tensor is not None:
        tensor.data.normal_()




def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a

'''
def cconv(a, b):
    return torch.irfft(com_mult(torch.rfft(a, 1), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


def ccorr(a, b):
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))
'''
def cconv(a, b):
    return irfft(com_mult(rfft(a, 1), rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


def ccorr(a, b):
    return irfft(com_mult(conj(rfft(a, 1)), rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))
