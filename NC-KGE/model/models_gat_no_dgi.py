from helper import *
import torch.nn as nn
from utils.func import *
from model.dgi_sig import DGI_Sig
from torch_scatter import scatter_mean, scatter_softmax, scatter_add
from model.GAT import *
import copy
import torch
import numpy as np
import shelve


class BaseModel(torch.nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.p = params
        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()
        

        self.x_bn = torch.nn.BatchNorm1d(self.p.num_ent)
        self.x_ln = torch.nn.LayerNorm(self.p.num_ent)

        if self.p.conve_act == 'relu':  
            self.conve_act = nn.ReLU() 
        elif self.p.conve_act == 'gelu':
            self.conve_act  = nn.GELU()
        elif self.p.conve_act == 'tanh':
            self.conve_act = nn.Tanh()
        elif self.p.conve_act == 'leakyrelu':    
            self.conve_act = nn.LeakyReLU(0.1)
        elif self.p.conve_act == 'selu': 
            self.conve_act = nn.SELU()
        elif self.p.conve_act == 'hardswish':
            self.conve_act = nn.Hardswish()  
        else:
            self.conve_act = nn.ReLU6()    
        #

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)
    

class MMGCNBase(BaseModel):
    def __init__(self, device, nhid1, dropout, hid_units, non_linearity, params=None):
        super(MMGCNBase, self).__init__(params)
        
        self.node_bn = torch.nn.BatchNorm1d(200)
        self.node_ln = torch.nn.LayerNorm(200)

        self.re_bn = torch.nn.BatchNorm1d(200)
        self.re_ln = torch.nn.LayerNorm(200)

        self.device = device
        self.S_GAT = GAT(params.num_of_layers,
                         params.num_heads_per_layer,
                         params.num_features_per_layer,
                         params.add_skip_connection,
                         params.bias_gat,
                         params.dropout_gat,
                         False)
        self.dropout = dropout
        self.num_nodes = params.num_ent
        self.num_relation = params.num_rel
        self.DGI = DGI_Sig(200, hid_units, non_linearity)
        self.marginloss = nn.MarginRankingLoss(0.5)
        # input-relation-multi-modal-feature
        self.relation_embeddings_ = nn.Parameter(torch.zeros(
            size=(2*self.num_relation, 100)))
        nn.init.xavier_uniform_(self.relation_embeddings_.data, gain=1.414)


        if not self.p.use_pretrain:
            self.entity_embeddings_ = nn.Parameter(torch.zeros(size=(self.num_nodes, 200)))
            nn.init.xavier_uniform_(self.entity_embeddings_.data, gain=1.414)
        else:
            self.node_init_emb = self.pretrian_node_emb()
            self.node_init_w = torch.nn.Linear(self.node_init_emb.size(1), 200, bias=False).cuda() 
            self.entity_embeddings_ = None
        
        self.layer_emb = nn.Sequential(
            nn.Linear(400, 1),
        )
        self.rel_emb = nn.Sequential(
            nn.Linear(100, 200),
        )
        self.sub_layer_emb = nn.Sequential(
            nn.Linear(400, 200),
        )
        self.layer_emb_out = nn.Sequential(
            nn.Linear(400, 200),
            nn.Dropout(self.dropout)
        )
        self.b_x, self.b_node_graph_index, self.b_edge_index, \
            self.b_new_adj, self.edge_graph_index, self.adj_edge = batch_graph_gen(params)
        self.m = nn.Softmax(dim=1)
        self.bn = torch.nn.BatchNorm1d(200)
        # self.big_adj = load_graph(params)
        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
    ###################################################

    ###################################################
    def multi_context_encoder(self, entity_feat, rel_feat):
        new_entity_rel_embed = entity_feat[self.b_x]
        edge_list = torch.LongTensor(self.adj_edge).to(self.device)
        data = [new_entity_rel_embed.cuda(), edge_list.cuda()]
        entity_embed, _ = self.S_GAT(data)
        index = torch.tensor(self.b_x).long().cuda()
        out = scatter_mean(entity_embed, index, dim=0)
        z = out[index]
        emb = torch.cat([entity_embed[index], z], dim=-1)
        new_emb = self.layer_emb(emb)
        z_s = scatter_softmax(new_emb, index, dim=0)
        new_out = scatter_add(z_s * emb, index, dim=0)

        if self.p.dataset == 'New_PharmKG8k-28':
            new_out = torch.cat([self.layer_emb_out(new_out[:]) + entity_feat[:7520], entity_feat[7520:]], dim=0) 
        elif self.p.dataset == '50P_new_DRKG':
            new_out = torch.cat([self.layer_emb_out(new_out[:]) + entity_feat[:31254], entity_feat[31254:]], dim=0) 
        elif self.p.dataset == '75P_new_DRKG':
            new_out = torch.cat([self.layer_emb_out(new_out[:]) + entity_feat[:34968], entity_feat[34968:]], dim=0)
        elif self.p.dataset == '100P_new_DRKG':
            new_out = torch.cat([self.layer_emb_out(new_out[:]) + entity_feat[:37936], entity_feat[37936:]], dim=0)
        elif self.p.dataset == 'DRKG17k-21':
            new_out = torch.cat([self.layer_emb_out(new_out[:]) + entity_feat[:16141], entity_feat[16141:]], dim=0)
        elif self.p.dataset == 'DB8k-28':
            new_out = torch.cat([self.layer_emb_out(new_out[:]) + entity_feat[:8256], entity_feat[8256:]], dim=0)
        elif self.p.dataset == 'BioKG72k_14':
            new_out = torch.cat([self.layer_emb_out(new_out[:]) + entity_feat[:70108], entity_feat[70108:]], dim=0)
        else: raise Exception('dataset name not exit, please set here')

        new_rel_embed = rel_feat[self.b_node_graph_index]
        rel_index = torch.tensor(self.b_node_graph_index).long().cuda()
        rel_embeds = scatter_mean(new_rel_embed, rel_index, dim=0)
 
        new_rel = torch.cat([rel_embeds, rel_embeds], dim=0)
        new_rel_out = self.rel_emb(new_rel) + self.rel_emb(rel_feat)
        if self.p.loss_function == 'contrastive_loss' or self.p.loss_function == 'magr_loss':
            new_out = self.node_ln(new_out) 
            new_rel_out = self.re_ln(new_rel_out) 
        
        return new_out, new_rel_out, entity_embed, emb

    def sub_contrast(self, sub_entity_embed, sub_rel):
        sub_rel = self.sub_layer_emb(sub_rel)
        shuf_index = torch.randperm(sub_rel.size(0))
        sub_entity_embed_ = sub_entity_embed[shuf_index]
        sub_rel_ = sub_rel[shuf_index]
        logits_aa = torch.sigmoid(torch.sum(sub_entity_embed * sub_rel, dim=-1))
        logits_bb = torch.sigmoid(torch.sum(sub_entity_embed_ * sub_rel_, dim=-1))
        logits_ab = torch.sigmoid(torch.sum(sub_entity_embed * sub_rel_, dim=-1))
        logits_ba = torch.sigmoid(torch.sum(sub_entity_embed_ * sub_rel, dim=-1))
        TotalLoss = 0.0
        ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
        TotalLoss += self.marginloss(logits_aa, logits_ba, ones)
        TotalLoss += self.marginloss(logits_bb, logits_ab, ones)
        return TotalLoss
    ###################################################
    def forward_base(self, sub, rel, drop):
        if self.p.use_pretrain: 
            self.entity_embeddings_ = torch.tanh(self.node_init_w(self.node_init_emb))
            
        entity_feat = self.entity_embeddings_
        rel_feat = self.relation_embeddings_
        entity_con, rel_con, sub_entity_embed, sub_rel= self.multi_context_encoder(entity_feat, rel_feat)
        cl_loss = self.sub_contrast(sub_entity_embed, sub_rel)
        entity_con = drop(entity_con)
        sub_emb = torch.index_select(entity_con, 0, sub)
        rel_emb = torch.index_select(rel_con, 0, rel)

        return sub_emb, rel_emb, entity_con, cl_loss, rel_con


class GAT_TransE(MMGCNBase):
    def __init__(self, device, nhid1, dropout, hid_units, non_linearity, params=None):
        super(self.__class__, self).__init__(device, nhid1, dropout, hid_units, non_linearity, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent, cl_loss, all_rel = self.forward_base(sub, rel, drop=self.drop)
        obj_emb = sub_emb + rel_emb

        x = self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
        if self.p.loss_function == 'contrastive_loss':
            score = x  
            score = self.x_ln(score) 
        else:
            score = x 
            
        return score, cl_loss

class GAT_DistMult(MMGCNBase):
    def __init__(self, device, nhid1, dropout, hid_units, non_linearity, params=None):
        super(self.__class__, self).__init__(device, nhid1, dropout, hid_units, non_linearity, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent, cl_loss, all_rel = self.forward_base(sub, rel, drop=self.drop)
        obj_emb = sub_emb * rel_emb

        x = torch.mm(obj_emb, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        if self.p.loss_function == 'contrastive_loss':
            score = x  
            score = self.x_ln(score) 
        else:
            score = x 
            
        return score, cl_loss
    


class GAT_ComplEx(MMGCNBase):
    def __init__(self, device, nhid1, dropout, hid_units, non_linearity, params=None):
        super(self.__class__, self).__init__(device, nhid1, dropout, hid_units, non_linearity, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent, cl_loss, all_rel = self.forward_base(sub, rel, drop=self.drop)

        heads = sub_emb
        tails = all_ent
        rels  = rel_emb
        
        heads_re, heads_im = torch.chunk(heads, chunks=2, dim=-1)#分离实部与虚部
        tails_re, tails_im = torch.chunk(tails, chunks=2, dim=-1)
        rels_re, rels_im   = torch.chunk(rels, chunks=2, dim=-1)

        x = (
                 torch.mm(rels_re * heads_re, tails_re.transpose(1, 0)) +
                 torch.mm(rels_re * heads_im, tails_im.transpose(1, 0)) +
                 torch.mm(rels_im * heads_re, tails_im.transpose(1, 0))-
                 torch.mm(rels_im * heads_im, tails_re.transpose(1, 0))
                 )

        x += self.bias.expand_as(x)

        if self.p.loss_function == 'contrastive_loss':
            score = x  
            score = self.x_ln(score) 
        else:
            score = x 
            
        return score, cl_loss
    


class GAT_SimplE(MMGCNBase):
    def __init__(self, device, nhid1, dropout, hid_units, non_linearity, params=None):
        super(self.__class__, self).__init__(device, nhid1, dropout, hid_units, non_linearity, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent, cl_loss, all_rel = self.forward_base(sub, rel, drop=self.drop)

        heads = sub_emb
        tails = all_ent
        rels  = rel_emb

        heads_h, heads_t = torch.chunk(heads, chunks=2, dim=-1)
        tails_h, tails_t = torch.chunk(tails, chunks=2, dim=-1)
        rel_a, rel_b = torch.chunk(rels, chunks=2, dim=-1)

        x = (
                 torch.mm(heads_h * rel_a, tails_t.transpose(1, 0)) +
                 torch.mm(heads_t * rel_b, tails_h.transpose(1, 0))
                 ) / 2
     
        x += self.bias.expand_as(x)
        
        if self.p.loss_function == 'contrastive_loss':
            score = x  
            score = self.x_ln(score) 
        else:
            score = x 
            
        return score, cl_loss




class GAT_RotatE(MMGCNBase):
    def __init__(self, device, nhid1, dropout, hid_units, non_linearity, params=None):
        super(self.__class__, self).__init__(device, nhid1, dropout, hid_units, non_linearity, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel):
        #sub_emb, rel_emb, all_ent, cl_loss = self.forward_base(sub, rel, drop=self.drop)
        head, relation, tail, cl_loss, all_rel = self.forward_base(sub, rel, drop=self.drop)
        mode = 0
        
        head, relation, tail = head.unsqueeze(dim=1), relation.unsqueeze(dim=1), tail.unsqueeze(dim=0)
        tail = tail.repeat(head.size(0), 1, 1)
        
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch': 
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:  
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)
        score = self.gamma.item() - score.sum(dim = 2)    
        
        
        x = score
        if self.p.loss_function == 'contrastive_loss':
            score = x  
            score = self.x_ln(score) 
        else:
            score = x 
        return score, cl_loss
    


class GAT_RotatEv2(MMGCNBase):
    def __init__(self, device, nhid1, dropout, hid_units, non_linearity, params=None):
        super(self.__class__, self).__init__(device, nhid1, dropout, hid_units, non_linearity, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)
        
        self.epsilon = 2.0
        self.nentity = self.num_nodes
        self.nrelation = self.num_relation
        self.hidden_dim = self.p.embed_dim
        
        self.gamma = nn.Parameter(
            torch.Tensor([self.p.gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.hidden_dim]), 
            requires_grad=False
        )

    def forward(self, sub, rel):
        #sub_emb, rel_emb, all_ent, cl_loss = self.forward_base(sub, rel, drop=self.drop)
        head, relation, tail, cl_loss, all_rel = self.forward_base(sub, rel, drop=self.drop)
        mode = 0
        
        head, relation, tail = head.unsqueeze(dim=1), relation.unsqueeze(dim=1), tail.unsqueeze(dim=0)
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        #Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_relation_head, re_relation_tail = torch.chunk(re_relation, 2, dim=-1)
        im_relation_head, im_relation_tail = torch.chunk(im_relation, 2, dim=-1)
        
        
        re_score_head = re_head * re_relation_head - im_head * im_relation_head
        im_score_head = re_head * im_relation_head + im_head * re_relation_head

        re_score_tail = re_tail * re_relation_tail - im_tail * im_relation_tail
        im_score_tail = re_tail * im_relation_tail + im_tail * re_relation_tail

        re_score = re_score_head - re_score_tail
        im_score = im_score_head - im_score_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)
        score = self.gamma.item() - score.sum(dim = -1)   
        
        
        x = score
        if self.p.loss_function == 'contrastive_loss':
            score = x  
            score = self.x_ln(score) 
        else:
            score = x 
        return score, cl_loss
        


    
class GAT_PaiRE(MMGCNBase):
    def __init__(self, device, nhid1, dropout, hid_units, non_linearity, params=None):
        super(self.__class__, self).__init__(device, nhid1, dropout, hid_units, non_linearity, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel):
        head, relation, tail, cl_loss, all_rel = self.forward_base(sub, rel, drop=self.drop)
        mode = 0
        
        print('head, relation, tail.shape:', head.shape, relation.shape, tail.shape)
        
        head, relation, tail = head.unsqueeze(dim=1), relation.unsqueeze(dim=1), tail.unsqueeze(dim=0)
        tail = tail.repeat(head.size(0), 1, 1)
        
        print('head, relation, tail.shape:', head.shape, relation.shape, tail.shape)
        #head, relation, tail.shape: torch.Size([512, 1, 200]) torch.Size([512, 1, 200]) torch.Size([512, 14541, 200])

        
        re_head, re_tail = torch.chunk(relation, 2, dim=-1)

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)

        score = head * re_head - tail * re_tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=-1)
        
        x = score
        if self.p.loss_function == 'contrastive_loss':
            score = x  
            score = self.x_ln(score) 
        else:
            score = x 
        return score, cl_loss
    


class GAT_ConvE(MMGCNBase):
    def __init__(self, device, nhid1, dropout, hid_units, non_linearity, params=None):
        super(self.__class__, self).__init__(device, nhid1, dropout, hid_units, non_linearity, params)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)
        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz),
                                       stride=1, padding=0, bias=self.p.bias)

        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)
        


    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.p.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.p.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent, cl_loss, all_rel = self.forward_base(sub, rel, drop=self.hidden_drop)
        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        # x= self.inp_drop(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        #x = F.relu(x)
        x = self.conve_act(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        #x = F.relu(x) 
        x = self.conve_act(x)
        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        if self.p.loss_function == 'contrastive_loss' or self.p.loss_function == 'magr_loss':
            score = self.x_ln(x) 
        else:
            score = x 

        if self.p.store_emb != None:
            return score, cl_loss, all_ent, all_rel
        else:
            return score, cl_loss
