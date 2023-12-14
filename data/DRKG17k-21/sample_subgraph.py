import sys
import math
from tqdm import tqdm
import time
from collections import defaultdict
from ordered_set import OrderedSet
import copy
import difflib
import Levenshtein
import joblib
import os
import random
from collections import Counter
import math
import numpy


def sample():
    #采样存在节点对之间有多个关系的子图
    
    file = 'train.txt'
    node2neighbor_triple    = defaultdict(list)
    node2rel                = defaultdict(list) #用元组(node, rel)作为关键词
    hnode2tnode             = defaultdict(list) #头节点到尾节点
    with open(file, 'r')as f:
        for i in f:
            tg = i.strip('\n').split('\t')
            node2neighbor_triple[tg[0]].append(i.strip('\n'))
            node2rel[(tg[0], tg[2])].append(tg[1])
            hnode2tnode[tg[0]].append(tg[2])
    
    exit_node = []
    with open('example_subgraph.txt', 'w')as f:
        for k in node2rel:
            if len(node2rel[k]) >= 2 and k[0] not in exit_node:
                node = k[0]
                
                for t in hnode2tnode[node]:   #继续判断是否还存在其它的节点对之间有多个关系的情况
                    if t == k[1]:
                        continue
                    if len(node2rel[(node, t)]) >= 2:
                        exit_node.append(node)
                        for triple in node2neighbor_triple[node]:
                            f.write(triple + '\n')

                        break
                    else:
                        continue
        
            
if __name__ == '__main__':
    sample()