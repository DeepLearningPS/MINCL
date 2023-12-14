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
import networkx as nx
import networkx as nx  # 导入 NetworkX 工具包
import matplotlib.pyplot as plt


def graph_count(file = None):
    #统计图数据的一些基本信息
    # 创建 图
    G1 = nx.Graph()  # 创建：空的 无向图。我们在处理数据的时候是按双向来的，即考虑逆关系，因此节点的出入度是一样的，所以统计出入度时按无向图来即可
    #G2 = nx.DiGraph()  #创建：空的 有向图
    #G3 = nx.MultiGraph()  #创建：空的 多图
    #G4 = nx.MultiDiGraph()  #创建：空的 有向多图
    
    with open(file, 'r') as f:
        for h,r,t in f.strip('\n').split('\t'):
            # 边(edge)的操作
            G1.add_edge(1,5)  # 向 G1 添加边 1-5，并自动添加图中没有的顶点
            
    
    

    nx.draw_networkx(G1)
    plt.show()
    plt.savefig("demo1.png") # 将图片保存到文件中
    

    
    nx.info(G1)	            #返回图的基本信息
    nx.degree(G1)           #返回图中各顶点的度
    nx.degree_histogram(G1) #节点图的直方图
    nx.pagerank(G1)         #返回图中各顶点的频率分布
    
    
    if __name__ == "__main__":
        file = 'train.txt'
        graph_count(file)