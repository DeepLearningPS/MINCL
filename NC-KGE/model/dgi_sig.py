import torch
import torch.nn as nn
from layer.readout import AvgReadout
from layer.discriminator import Discriminator
import torch
import torch.nn.functional as F
import numpy as np
import random
class DGI_Sig(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI_Sig, self).__init__()
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.dropout = 0.1
        self.disc = Discriminator(n_in)
        # self.MLP_CL1 = nn.Sequential(
        #     nn.Linear(200, 400),
        #     nn.PReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(400, 200)
        # )

    def forward(self, seq1_top, seq2_top, sparse, msk, samp_bias1, samp_bias2):
        # topology
        # h_1_top_node = self.gcn(seq1_top, adj.cuda(), sparse)
        h_1_top_node = seq1_top
        # h_1_top_node_ = self.MLP_CL1(h_1_top_node)
        c_top_graph = self.read(h_1_top_node, msk)
        c_top_graph = self.sigm(c_top_graph)
        # h_2_top_node = self.gcn(seq2_top, adj.cuda(), sparse)
        h_2_top_node = seq2_top
        # h_2_top_node_ = self.MLP_CL1(h_2_top_node)

        ret = self.disc(c_top_graph, h_1_top_node, h_2_top_node, samp_bias1, samp_bias2)
        return ret
