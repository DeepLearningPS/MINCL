from datetime import datetime

import torch
import os
import pickle5 as pickle
import numpy as np
import tf_geometric as tfg
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm
import random


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def gen_adj_S(sample_struct_edges, values):
    shape_size = max([max(sample_struct_edges[:, 0]), max(sample_struct_edges[:, 1])]) + 1
    sedges = np.array(list(sample_struct_edges), dtype=np.int32).reshape(
        sample_struct_edges.shape)  
    sadj = sp.coo_matrix((values, (sedges[:, 0], sedges[:, 1])),
                         shape=(shape_size, shape_size),
                         dtype=np.float32)
    nsadj = sparse_mx_to_torch_sparse_tensor(sadj)
    return nsadj


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def gen_adj(sample_struct_edges):
    shape_size = max([max(sample_struct_edges[:, 0]), max(sample_struct_edges[:, 1])]) + 1
    sedges = np.array(list(sample_struct_edges), dtype=np.int32).reshape(
        sample_struct_edges.shape)  
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
                         shape=(shape_size, shape_size),
                         dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)  
    nsadj = normalize(sadj + sp.eye(sadj.shape[0]))  
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    return nsadj


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
    for i, item in enumerate(rel_entity_set):
        new_entity2id[i] = {}
        for j, _ in enumerate(item):
            new_entity2id[i][_] = j
    
    new_rel_dic = {}
    for key, tris in rel_dic.items():
        temp = []
        for tri in tris:
            _ = [new_entity2id[key][tri[0]], new_entity2id[key][tri[1]]]
            temp.append(_)
        new_rel_dic[key] = np.array(temp)
    return new_entity2id, new_rel_dic


def get_new_entity2id(args):
    file = '../data/' + args.dataset + "/new_1hop.pickle"
    with open(file, 'rb') as handle:
        new_1hop = pickle.load(handle)
    file = '../data/' + args.dataset + "/rel_dic.pickle"
    with open(file, 'rb') as handle:
        rel_dic = pickle.load(handle)
    new_entity2id, new_rel_dic = specify_node_update(rel_dic)
    return new_entity2id


def init_embeddings(entity_file, relation_file):
    with open(entity_file, 'rb') as handle:
        entity_emb = pickle.load(handle)
    with open(relation_file, 'rb') as handle:
        relation_emb = pickle.load(handle)
    return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)


def norm_embeddings(embeddings: torch.Tensor):
    norm = embeddings.norm(p=2, dim=-1, keepdim=True)
    return embeddings / norm


def load_graph(args):
    featuregraph_path = '../data/' + args.dataset + '/edges_train.txt'
    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])),
                         shape=(args.num_ent, args.num_ent),
                         dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    return nfadj


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


def batch_graph_gen(args): 
    new_entity2id = get_new_entity2id(args)
    edge_index = read_edge_index(args)
    big_graph = []
    entity_index = []
    for key, val in new_entity2id.items():
        temp = []
        for k in val.keys():
            temp.append(k)
        entity_index.append(temp)

    adj_edge = np.loadtxt( '../data/' + args.dataset + "/batchgraph_edgeindex.txt")  
    adj_edge = adj_edge.astype(int)
    
    
    for i in range(len(new_entity2id)):
        sub_graph = tfg.Graph(x=entity_index[i], edge_index=edge_index[i])
        big_graph.append(sub_graph)
    batch_graph = tfg.BatchGraph.from_graphs(big_graph)
    new_edges = []
    a = []
    edges_data = batch_graph.edge_index
    for i in range(len(edges_data[0])):
        a.append(edges_data[0][i])
        new_edges.append([edges_data[0][i], edges_data[1][i]])
    print('len(a):', len(a))
    new_adj = gen_adj((np.array(new_edges)))
    return (batch_graph.x), (batch_graph.node_graph_index), (batch_graph.edge_index), (
        new_adj), (batch_graph.edge_graph_index), (adj_edge)


def read_edge_index(args):
    file = '../data/' + args.dataset + "/edge_index3.pickle"
    with open(file, 'rb') as handle:
        edge_index = pickle.load(handle)
    return edge_index


def gen_shuf_fts(entity):
    idx = np.random.permutation(len(entity))
    
    shuf_fts = entity[idx, :] + torch.randn(entity.shape).cuda()  
    return shuf_fts
