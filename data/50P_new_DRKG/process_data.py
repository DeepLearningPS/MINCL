'''
所有数据一键生成

'''
import os
import numpy as np
import re
import pprint
from collections import OrderedDict#python3.6之后就有顺序了
from collections import defaultdict
import seaborn
import matplotlib.pyplot as plt
import matplotlib
import seaborn
from ordered_set import OrderedSet
import pickle

import torch
import random
import numpy as np

torch.cuda.manual_seed_all(47)
torch.manual_seed(48)
np.random.seed(49)
random.seed(50)

#要保证之后生成的所有实体与关系id与全局id(即relation2id.txt,entity2id.txt)保持一致，不要使用局部id

def gen_id(file_list): #生成所有实体名和关系名到id
    relation_set = OrderedSet()
    entity_set = OrderedSet()
    for file in file_list:
        with open(file) as f:
            for i in f:
                tg = i.rstrip('\n').split('\t')
                if len(tg) != 3:
                    raise Exception('len(tg) != 3')
                relation_set.add(tg[1])
                entity_set.add(tg[0])
                entity_set.add(tg[2])

    print('len(relation_set):',len(relation_set))
    print('len(entity_set):',len(entity_set))
    
    with open('relation2id.txt','w')as f:
        for ids,tg in enumerate(relation_set):
            f.write(tg + '\t' + str(ids) + '\n')
            
    with open('entity2id.txt','w')as f:
        for ids,tg in enumerate(entity_set):
            f.write(tg + '\t' + str(ids) + '\n')
            

def gen_split_id(file_list): #生成各个数据分割的实体名到id
    all_entity2id = {}
    with open('entity2id.txt')as f:
        for i in f:
            tg = i.rstrip('\n').split('\t')
            all_entity2id[tg[0]] = tg[1]
    print('len(all_entity2id):',len(all_entity2id))
    
        
    entity_set = OrderedSet()
    with open(file_list[0]) as f:
        for i in f:
            tg = i.rstrip('\n').split('\t')
            entity_set.add(tg[0])
            entity_set.add(tg[2])
            
    with open('train_entity2id.txt','w')as f:
        for entity in entity_set:
            ids = all_entity2id[entity]
            f.write(entity + '\t' + str(ids) + '\n')
            
    print('len(entity_set):',len(entity_set))




    entity_set = OrderedSet()
    with open(file_list[1]) as f:
        for i in f:
            tg = i.rstrip('\n').split('\t')
            entity_set.add(tg[0])
            entity_set.add(tg[2])
            
    with open('valid_entity2id.txt','w')as f:
        for entity in entity_set:
            ids = all_entity2id[entity]
            f.write(entity + '\t' + str(ids) + '\n')

    print('len(entity_set):',len(entity_set))




    entity_set = OrderedSet()
    with open(file_list[2]) as f:
        for i in f:
            tg = i.rstrip('\n').split('\t')
            entity_set.add(tg[0])
            entity_set.add(tg[2])
            
    with open('test_entity2id.txt','w')as f:
        for entity in entity_set:
            ids = all_entity2id[entity]
            f.write(entity + '\t' + str(ids) + '\n')

    print('len(entity_set):',len(entity_set))




def gen_tri_id(): #生成三元组id形式
    entity_dic = {}
    file = "entity2id.txt"
    with open(file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip().split("\t")
            entity_dic[line[0]] = line[1]


    rel_dic= {}
    file = "relation2id.txt"
    with open(file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip().split("\t")
            rel_dic[line[0]] = line[1]

    
    with open("train.txt", "r", encoding="utf-8") as f_t:
        with open("train2id.txt", "w", encoding="utf-8") as f_w: #将训练集变成全id
            for line in f_t.readlines():
                line = line.strip().split("\t")
                temp = str(entity_dic[line[0]] ) + "\t" + str(rel_dic[line[1]]) + "\t" + str(entity_dic[line[2]]) + "\n"
                f_w.write(temp) #我们改成以\t为间隔符



    with open("valid.txt", "r", encoding="utf-8") as f_t:
        with open("valid2id.txt", "w", encoding="utf-8") as f_w: #将验证集变成全id
            for line in f_t.readlines():
                line = line.strip().split("\t")
                temp = str(entity_dic[line[0]] ) + "\t" + str(rel_dic[line[1]]) + "\t" + str(entity_dic[line[2]]) + "\n"
                f_w.write(temp) #我们改成以\t为间隔符



    with open("test.txt", "r", encoding="utf-8") as f_t:
        with open("test2id.txt", "w", encoding="utf-8") as f_w: #将测试集变成全id
            for line in f_t.readlines():
                line = line.strip().split("\t")
                temp = str(entity_dic[line[0]] ) + "\t" + str(rel_dic[line[1]]) + "\t" + str(entity_dic[line[2]]) + "\n"
                f_w.write(temp) #我们改成以\t为间隔符

 
    
def gen_edges():  #生成节点对
    entity_dic = {}
    file = "entity2id.txt"
    with open(file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip().split("\t")
            entity_dic[line[0]] = line[1]

    with open("train.txt", 'r', encoding="utf-8") as f:
        with open("edges_train.txt", 'w', encoding="utf-8") as f_w:
            for line in f.readlines():
                line = line.strip().split("\t")
                temp = entity_dic[line[0]] + "\t" + entity_dic[line[2]] + "\n"
                f_w.write(temp)#生成训练集使用的头尾节点对（h,t),以id的形式，这里没考虑自环和逆关系



def gen_train_entity(): #生成训练集的实体id
    entity_dic = {}
    file = "entity2id.txt"
    with open(file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip().split("\t")
            entity_dic[line[0]] = line[1]
    
    u_entity = OrderedSet()

    with open("train.txt", "r", encoding="utf-8") as f:
        for i in f:
            tg = i.rstrip('\n').split('\t')
            u_entity.add(entity_dic[tg[0]])
            u_entity.add(entity_dic[tg[2]])
            

    with open("unique_entity_train.txt", "w", encoding="utf-8") as f:
        for item in u_entity:
            temp = str(item)+"\n"
            f.write(temp)


def get_batch_nhop_neighbors_all():
    batch_source_triples_1hop = []
    with open("train2id.txt", 'r', encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip().split("\t")
            temp = [int(line[0]), int(line[1]), int(line[2])]
            batch_source_triples_1hop.append(temp)
    #return np.array(batch_source_triples_1hop).astype(np.int32) #二维数组
    return batch_source_triples_1hop #二维list,下面的遍历时，list速度更快

    '''
    #由于使用多阶多跳，所以这里采样每阶的子图（即三元组），这里的邻居采样比例对模型影响很大，如果只使用一阶，则全部采样，即整个训练集
    #这里没有考虑自环和逆关系？看看train2id.txt文件？原始的三元组，没有自环和逆边

    '''

def readfile(): #获取训练集所有节点id
    unique_entities_train = []
    with open("unique_entity_train.txt", 'r', encoding="utf-8") as f:
        for line in f.readlines():
            line = int(line.strip())
            unique_entities_train.append(line)
    return unique_entities_train


def gen_new_xhop3(batch_source_triples_1hop, unique_entities_train):
    
    new_1hop2 = {}
    for entity_u in unique_entities_train:#遍历每一个训练实体。
        temp = []
        # print(i)
        for i in batch_source_triples_1hop: #遍历每一个三元组
            if entity_u == i[0] or entity_u == i[2]: 
                temp.append(i)
        new_1hop2[entity_u] = temp
    return new_1hop2
    #上述训练太耗时间了
    
    

    '''
    上面的2层循环的意思是：找出所有以该实体的为头尾节点的三元组
    实际上是获取了一跳和二跳邻居，注意这不同于逆关系和自环。由于知识图谱是个有向图，所以要考虑逆边。图一般按有向图处理，
    无向图的边一样，有向图的边则增加逆边和自环，此时无论哪种情况，三元组的数量要比原来多出一倍多，这在信息聚合的时候可以
    方便一次性获取所有节点的嵌入。
    {entity_id:[[]]}

    '''

def gen_new_1hop(): #生成实体对应的一跳二跳邻居，形式是二维list组成的字典
    batch_source_triples_1hop = get_batch_nhop_neighbors_all()#获取1跳邻居
    # file = "1hop.pickle"
    # with open(file, 'wb') as handle:
    #     pickle.dump(batch_source_triples_1hop, handle,
    #                 protocol=pickle.HIGHEST_PROTOCOL)
    # file = "1hop.pickle"
    # with open(file, 'rb') as handle:
    #         batch_source_triples_1hop = pickle.load(handle)

    unique_entities_train = readfile()
    new_1hop = gen_new_xhop3(batch_source_triples_1hop, unique_entities_train)#使用2跳邻居
    file = "new_1hop.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(new_1hop, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)



def gen_intra_context_edges(new_1hop): #new_1hop,每一个实体对应的正逆向三元组，字典的形式
    rel_dic = {}
    rel_dic2 = {}
    relation_num = 237
    for i in range(relation_num): #关系数量根据数据集而变
        #print(i)
        temp = []
        for tris in new_1hop.values(): #实体的一跳，二跳邻居
            for tri in tris: # tris, 二维list
                if tri[1] == i and [tri[0], tri[-1]] not in temp: #找每一种关系对应的三元组
                    temp.append([tri[0], tri[-1]])
        rel_dic[i] = np.array(temp) #存放关系对应的三元组，{re_id:[[]]}，这里不建议变成np数组，遍历太慢，后期还要转
        rel_dic2[i] = temp #保存list较好
    return rel_dic,rel_dic2


def gen_relation_triple(): #生成关系对应的三元组（包括二跳三元组）

    file = "new_1hop.pickle"
    with open(file, 'rb') as handle:
        new_1hop = pickle.load(handle)
        
    rel_dic,rel_dic2 = gen_intra_context_edges(new_1hop)
    file = "rel_dic.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(rel_dic, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


    file = "rel_dic2.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(rel_dic2, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


def specify_node_update(rel_dic):
    rel_entity_set = []
    for tris in rel_dic.values():
        temp = []
        for tri in tris:
            if tri[0] not in temp:
                temp.append(tri[0])
            if tri[1] not in temp:
                temp.append(tri[1])
        rel_entity_set.append(temp)
    new_entity2id = {}
    for i, item in enumerate(rel_entity_set):  # 关系实体集合：r1:{e1,e2,e3......}
        new_entity2id[i] = {}
        for j, _ in enumerate(item):
            new_entity2id[i][_] = j  # 对于一种关系里面的实体集合，遍历实体，对每个实体重新编号[关系r1][实体e]:编号j，形成键值对，实体：编号
    # update new subgraph entity id
    new_rel_dic = {}
    for key, tris in rel_dic.items():
        temp = []
        for tri in tris:
            _ = [new_entity2id[key][tri[0]], new_entity2id[key][tri[1]]]  # 从重编号的new_entity2id里面取值，[关系r1][实体e]
            temp.append(_)
        new_rel_dic[key] = np.array(temp)  # 新的关系集：关系：(重新编号头实体，重新编号尾实体)
    return new_entity2id, new_rel_dic


def gen_adj(sample_struct_edges, new_entity2id):
    sedges = np.array(list(sample_struct_edges), dtype=np.int32).reshape(sample_struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
                         shape=(len(new_entity2id), len(new_entity2id)),
                         dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = self.normalize(sadj + sp.eye(sadj.shape[0]))
    nsadj = self.sparse_mx_to_torch_sparse_tensor(nsadj)
    return nsadj


def edge_index_gen(new_entity2id, new_1hop):
    edge_index = []
    count = 0
    for key, val in new_entity2id.items():
        #print(count)
        count += 1
        ori = []
        dst = []
        entity_new_id = []
        entity_ori_id = []
        for tri_key, tri_val in val.items():
            entity_new_id.append(tri_val)
            entity_ori_id.append(tri_key)
        if len(entity_new_id) == 2:
            edge_index.append(np.array([[entity_new_id[0]], [entity_new_id[1]]]))
        else:
            temp_tris = []
            for key_, val_ in new_1hop.items():
                if key_ in entity_ori_id:
                    for tri in val_:
                        if tri[1] == key and tri not in temp_tris:
                            temp_tris.append(tri)
            for tri in temp_tris:
                ori.append(val[tri[0]])
                dst.append(val[tri[2]])
            edge_index.append(np.array([ori, dst]))
    file = "edge_index3.pickle" #节点对（看一下是为每一个子图单独生成还是全图就维持一个？？因为不仅要生成节点对，同时还要知道对于关系）
    with open(file, 'wb') as handle:
        pickle.dump(edge_index, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


def gen_subgraph(): # 这里的子图是因为GCN要处理异构关系，所以按关系生成同构子图
    file = "new_1hop.pickle"
    with open(file, 'rb') as handle:
        new_1hop = pickle.load(handle)
    file = "rel_dic.pickle"
    with open(file, 'rb') as handle:
        rel_dic = pickle.load(handle)
    new_entity2id, new_rel_dic = specify_node_update(rel_dic)
    edge_index_gen(new_entity2id, new_1hop)
    #multi_edge = gen_adj(new_rel_dic, new_entity2id)


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


def read_edge_index():
    file =  "edge_index3.pickle"
    with open(file, 'rb') as handle:
        edge_index = pickle.load(handle)
    return edge_index


def get_new_entity2id():
    file = "new_1hop.pickle"
    with open(file, 'rb') as handle:
        new_1hop = pickle.load(handle)
    file =  "rel_dic.pickle"
    with open(file, 'rb') as handle:
        rel_dic = pickle.load(handle)
    new_entity2id, new_rel_dic = specify_node_update(rel_dic)
    return new_entity2id


def batch_graph_gen(): #生成batchgraph_edgeindex.txt文件,该文件将在utils文件夹中的func.py文件中引用
    new_entity2id = get_new_entity2id()
    edge_index = read_edge_index()
    big_graph = []
    entity_index = []
    for key, val in new_entity2id.items():
        temp = []
        for k in val.keys():
            temp.append(k)
        entity_index.append(temp)
    rows = []
    cols = []
    for index, edge in enumerate(edge_index):
        for i in range(len(edge[0])):
            rows.append(get_key(new_entity2id[index], edge[0][i])[0])
            cols.append(get_key(new_entity2id[index], edge[1][i])[0])
    print(len(rows))
    
    adj_edge = [rows, cols]
    np.savetxt("batchgraph_edgeindex.txt", adj_edge)



def count(file_list):# 统计一下数据集的基本情况
    all_entity_set = OrderedSet()
    all_relation_set = OrderedSet()
    all_triple_set = OrderedSet()

    data_name = ['训练集','验证集','测试集']
    for file, data in zip(file_list,data_name):
        entity_set = OrderedSet()
        relation_set = OrderedSet()
        triple_set = OrderedSet()
        with open(file) as f:
            for i in f:
                tg = i.rstrip('\n').split('\t')
                tg = tuple(tg)
            
                entity_set.add(tg[0])
                entity_set.add(tg[2])
                relation_set.add(tg[1])
                triple_set.add(tg)
                
                all_entity_set.add(tg[0])
                all_entity_set.add(tg[2])
                all_relation_set.add(tg[1])
                all_triple_set.add(tg)
                
        print(data)       
        print('实体数量:',len(entity_set))
        print('关系数量:',len(relation_set))
        print('三元组数量:',len(triple_set))
        print('\n')

    
    print('总的实体数量:',len(all_entity_set))
    print('总的关系数量:',len(all_relation_set))
    print('总的三元组数量:',len(all_triple_set)) 


if __name__ == '__main__':
    file_list = ['train.txt','valid.txt','test.txt']
    
    gen_id(file_list)
    gen_split_id(file_list)
    
    gen_tri_id()
    gen_edges()
    gen_train_entity()
    gen_new_1hop()
    gen_relation_triple()
    gen_subgraph()
    
    
    batch_graph_gen()
    count(file_list)
    
