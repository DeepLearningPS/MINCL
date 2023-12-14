import sys
import math
from tqdm import tqdm
import time
from collections import defaultdict
import copy
import difflib
import Levenshtein
import joblib
import os
import random
from collections import Counter
import math
import numpy
import matplotlib
import seaborn
from ordered_set import OrderedSet
import pickle

import torch
import random
import numpy as np



def gen_special_testdata():
    os.makedirs('special_test', exist_ok=True)
    
    #思路，维持2个列表，一个是三元组，一个类型化的三元组，然后创建一个字典，字典的key是头尾节点的类型，value是其对应的三元组，最后保存字典即可 node2typeid.txt
    type2triple = defaultdict(list)
    node2type   = {}
    with open('node2type.txt')as f:
        for i in f:
            tg = i.strip('\n').split('\t')
            node2type[tg[0]] = tg[1]

    with open('test.txt')as f:
        for i in f:
            tg = i.strip('\n').split('\t')
            triple = i.strip('\n')
            type2triple[(node2type[tg[0]], node2type[tg[2]])].append(triple)
    
    
    new_type2triple = defaultdict(list)
    
    type_keys =  type2triple.keys()
    exit_keys = []
    
    for k in type2triple:
        print('k:', k)
        print('exit_keys:', exit_keys)
        print('================================')
        if (k[0], k[1]) in exit_keys:
            new_type2triple[(k[0], k[1])].extend(type2triple[k])
        elif (k[1], k[0]) in exit_keys:
            new_type2triple[(k[1], k[0])].extend(type2triple[k])
        else:
            new_type2triple[(k[0], k[1])].extend(type2triple[k])
            exit_keys.append(k)
                
    print('new_type2triple.keys():', new_type2triple.keys())        
    
    for type_name in new_type2triple:
        file_name = type_name[0] + '_' + type_name[1] + '.txt'
        with open(f'special_test/{file_name}', 'w') as f:
            for tri in new_type2triple[type_name]:
                f.write(tri + '\n')
                 
            

    

if __name__ == '__main__':
    gen_special_testdata()