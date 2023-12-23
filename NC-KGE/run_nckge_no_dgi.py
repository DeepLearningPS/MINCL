from helper import *
from data_loader import *
import sys

from model.models_gat_no_dgi import *
from tqdm import tqdm
import os
import time
import pprint



import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


local_rank=int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
torch.cuda.set_device(local_rank)

class Runner(object):

    def load_data(self):
        ent_set, rel_set = OrderedSet(), OrderedSet()
        train_node_set = set()
        for split in ['train', 'valid', 'test']:
            for line in open('../data/{}/{}.txt'.format(self.p.dataset, split)):
                
                sub, rel, obj = line.strip().split('\t') 
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)
                if split == 'train':
                    train_node_set.add(sub)
                    train_node_set.add(obj)

        print('train node num:', len(train_node_set))
        print('rel_set num:', len(rel_set))
        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for idx, rel in enumerate(rel_set)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}
        self.p.id2ent = self.id2ent
        self.p.id2rel = self.id2rel

        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

        self.data = ddict(list)
        sr2o = ddict(set)
        train_sr2o = ddict(set)

        if args.special_test != None: 
            for split in ['train', 'valid', 'test']:
                if split == 'test': 
                    for line in open('../data/{}/special_test/{}.txt'.format(self.p.dataset, args.special_test)):
                        
                        sub, rel, obj = line.strip().split('\t') 
                        sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                        self.data[split].append((sub, rel, obj))
                else:       
                    for line in open('../data/{}/{}.txt'.format(self.p.dataset, split)):
                        
                        sub, rel, obj = line.strip().split('\t') 
                        sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                        self.data[split].append((sub, rel, obj))

                    if split == 'train':
                        train_sr2o[(sub, rel)].add(obj)
                        train_sr2o[(obj, rel + self.p.num_rel)].add(sub)
        else:
            for split in ['train', 'valid', 'test']:
                for line in open('../data/{}/{}.txt'.format(self.p.dataset, split)):
                    
                    sub, rel, obj = line.strip().split('\t') 
                    sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                    self.data[split].append((sub, rel, obj))

                    if split == 'train':
                        train_sr2o[(sub, rel)].add(obj)
                        train_sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.data = dict(self.data)

        self.train_sr2o = {k: list(v) for k, v in train_sr2o.items()}
        
        for split in ['train', 'valid', 'test']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = ddict(list)
        
        
        
        if self.p.loss_function == 'bcel_loss':
            for (sub, rel), obj in self.train_sr2o.items(): 
                self.triples['train'].append({'triple': (sub, rel, -1), 'label': self.train_sr2o[(sub, rel)], 'sub_samp': 1})
        else:
            for split in ['train']:  
                for sub, rel, obj in self.data[split]:
                    rel_inv = rel + self.p.num_rel
                    self.triples['{}'.format(split)].append(
                        {'triple': (sub, rel, obj), 'label': self.train_sr2o[(sub, rel)], 'sub_samp': 1})
                    self.triples['{}'.format(split)].append(
                        {'triple': (obj, rel_inv, sub), 'label': self.train_sr2o[(obj, rel_inv)], 'sub_samp': 1})
         
        
        

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append(
                    {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append(
                    {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        self.triples = dict(self.triples)

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return DataLoader(
                dataset_class(self.triples[split], self.p),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.p.num_workers),
                collate_fn=dataset_class.collate_fn
            )

        self.data_iter = {
            'train': get_data_loader(TrainDataset, 'train', self.p.batch_size),
            'valid_head': get_data_loader(TestDataset, 'valid_head', self.p.batch_size),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail', self.p.batch_size),
            'test_head': get_data_loader(TestDataset, 'test_head', self.p.batch_size),
            'test_tail': get_data_loader(TestDataset, 'test_tail', self.p.batch_size),
        }

        self.edge_index, self.edge_type = self.construct_adj()

        if self.p.pretrained_emb:
            self.entity_embeddings, self.relation_embeddings = self.init_embeddings(
                os.path.join(self.p.dataset, 'entity2vec.txt'),
                os.path.join(self.p.dataset, 'relation2vec.txt'))
            print("Initialised relations and entities from TransE")
        else:
            self.entity_embeddings = np.random.randn(
                self.p.num_ent, 200)
            self.relation_embeddings = np.random.randn(
                self.p.num_rel, 200)
            print("Initialised relations and entities randomly")

    def init_embeddings(self, entity_file, relation_file):
        entity_emb, relation_emb = [], []

        with open('../data/' + entity_file) as f:
            for line in f:
                entity_emb.append([float(val) for val in line.strip().split()])

        with open('../data/' + relation_file) as f:
            for line in f:
                relation_emb.append([float(val) for val in line.strip().split()])

        return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)

    def construct_adj(self):
        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)

        edge_index = torch.LongTensor(edge_index).to(self.device).t()
        edge_type = torch.LongTensor(edge_type).to(self.device)

        return edge_index, edge_type

    def __init__(self, params):
        self.p = params
       
        self.logger = get_logger(args.task_name + '_'+ args.name + '.log', args.log_dir, args.config_dir)

        
        
        self.logger.setLevel(logging.DEBUG)
        
        
        for arg in vars(args):
            self.logger.info((arg, getattr(args, arg)))
        self.logger.info('\n\n')
        
        

        if self.p.gpu != '-1' and torch.cuda.is_available():
            
            self.device = torch.device('cuda')  
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data()  
        self.model = self.add_model(self.p.score_func)
            
        
        if self.p.lr_scheduler == 'CosineAnnealingLR': 
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = self.p.max_epochs)
            

        elif self.p.lr_scheduler == 'ReduceLROnPlateau': 
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min')
            

        elif self.p.lr_scheduler == 'CyclicLR': 
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.01, max_lr=0.1 )
 

        elif self.p.lr_scheduler == 'OneCycleLR': 
            self.scheduler =torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.01, \
                                                        steps_per_epoch = len(self.data_iter['train']), epochs = self.p.max_epochs)
      
        elif self.p.lr_scheduler == 'CosineAnnealingWarmRestarts': 
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0 = 10, T_mult = 1)


    def add_model(self, score_func):
        if score_func.lower() == 'transe':
            self.model = GAT_TransE(self.device, 400, 0.2, 512, non_linearity='prelu', params=self.p)
        elif score_func.lower() == 'distmult':
            self.model = GAT_DistMult(self.device, 400, 0.2, 512, non_linearity='prelu', params=self.p)
        elif score_func.lower() == 'conve':
           self. model = GAT_ConvE(self.device, 400, 0.2, 512, non_linearity='prelu', params=self.p)
        elif score_func.lower() == 'simple':
            self.model = GAT_SimplE(self.device, 400, 0.2, 512, non_linearity='prelu', params=self.p)
        elif score_func.lower() == 'complex':
            self.model = GAT_ComplEx(self.device, 400, 0.2, 512, non_linearity='prelu', params=self.p)
        elif score_func.lower() == 'rotate':
            self.model = GAT_RotatE(self.device, 400, 0.2, 512, non_linearity='prelu', params=self.p) 
        elif score_func.lower() == 'rotatev2':
            self.model = GAT_RotatEv2(self.device, 400, 0.2, 512, non_linearity='prelu', params=self.p) 
        elif score_func.lower() == 'paire':
            self.model = GAT_PaiRE(self.device, 400, 0.2, 512, non_linearity='prelu', params=self.p)
        else:
            raise NotImplementedError
            
        if self.p.save_path != None:
            print('retrain')
            
            model = self.model.cuda(local_rank)
            model = DDP(model, [local_rank], find_unused_parameters=True)
            model = model.module
            self.optimizer = self.add_optimizer(model.parameters())
            self.load_model(os.path.join(self.p.model_dir, self.p.save_path))
            num_param = sum([p.numel() for p in self.model.parameters()])
            self.logger.info("Number of parameters: {0}".format(num_param)) 
            
        else:
            print('train')
            
            
            model = self.model.cuda(local_rank)
            model = DDP(model, [local_rank], find_unused_parameters=True)
            model = model.module
            self.optimizer = self.add_optimizer(model.parameters())
            num_param = sum([p.numel() for p in model.parameters()])
            self.logger.info("Number of parameters: {0}".format(num_param)) 
            self.model = model
        
        
        return self.model

    def add_optimizer(self, parameters):
        if self.p.lr_scheduler != 'CyclicLR': 
            if args.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(parameters)
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(parameters, lr=0.001)
            elif args.optimizer == 'adagrad':
                optimizer = torch.optim.Adagrad(parameters)
        else:
            optimizer = torch.optim.SGD(parameters, lr=0.001)
            
        return optimizer

    def read_batch(self, batch, split):
        if split == 'train':
            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label
        else:
            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label

    def save_model(self, save_path):
        state = {
            'state_dict': self.model.state_dict(),
            'best_val': self.best_val,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args'	: vars(self.p)
        }
        torch.save(state, save_path)



    def save_model_epoch(self, save_path, epoch):
        state = {
            'state_dict': self.model.state_dict(),
            'best_val': self.best_val,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args'	: vars(self.p)
        }
        torch.save(state, save_path)



    def load_model(self, load_path):
        state			= torch.load(load_path, map_location=f'cuda:{local_rank}')
        state_dict		= state['state_dict']
        
        self.best_val		= state['best_val']
        self.best_val_mrr	= self.best_val['mrr']

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer']) 
        self.optimizer.param_groups[0]['lr'] = state['optimizer']['param_groups'][0]['lr']

    def evaluate(self, split):
        if split == 'valid':
            left_results,left_loss = self.predict(split=split, mode='tail_batch')
            right_results,right_loss = self.predict(split=split, mode='head_batch')
            valid_loss = (left_loss + right_loss) / 2
            results = get_combined_results(left_results, right_results)
            return results,valid_loss
        else:
            left_results = self.predict(split=split, mode='tail_batch')
            right_results = self.predict(split=split, mode='head_batch')
            results = get_combined_results(left_results, right_results)
            return results

    def predict(self, split='valid', mode='tail_batch'):
        self.model.eval()

        with torch.no_grad():
            results = {}
            losses = []
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

            for step, batch in enumerate(train_iter):

                if split == 'valid':
                    sub, rel, obj, label = self.read_batch(batch, 'valid')
                    pred, cl_loss = self.model.forward(sub, rel)
                    if self.p.loss_function == 'contrastive_loss':
                        tri_loss = self.contrastive_loss(pred, label, self.p.tau_plus, self.p.beta, self.p.estimator, obj)
                    elif self.p.loss_function == 'mask_softmax':
                        tri_loss = self.mask_softmax(pred, label, obj)
                    elif self.p.loss_function == 'mix_loss':
                        tri_loss = self.mix_loss(pred, label, self.p.tau_plus, self.p.beta, self.p.estimator, obj)
                    elif self.p.loss_function == 'bcel_loss':
                        tri_loss = self.bcel_loss(pred, label)
                    elif self.p.loss_function == 'magr_loss':
                        tri_loss = self.magr_loss(pred, label, obj)

                    loss = tri_loss + self.p.subgraph_loss_rate * cl_loss
                    losses.append(loss.item())
                else:
                    sub, rel, obj, label = self.read_batch(batch, split)
                    pred, cl_loss = self.model.forward(sub, rel)
                
                b_range = torch.arange(pred.size()[0], device=self.device)
                target_pred = pred[b_range, obj]
                pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]

                ranks = ranks.float()
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
                for k in range(10):
                    results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}'.format(k + 1), 0.0)

            if split == 'valid':
                valid_loss = np.mean(losses)
                return results,valid_loss
            else:
                return results




    def run_epoch(self, epoch, val_mrr = 0):
        self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train'])
        iters = len(self.data_iter['train'])
        b_xent = nn.BCEWithLogitsLoss() 
        lbl = torch.cat((torch.ones(1, self.p.num_ent), torch.zeros(1, self.p.num_ent)), 1).cuda() 
        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            sub, rel, obj, label = self.read_batch(batch, 'train')
            pred, cl_loss = self.model.forward(sub, rel)

            if self.p.loss_function == 'contrastive_loss':
                tri_loss = self.contrastive_loss(pred, label, self.p.tau_plus, self.p.beta, self.p.estimator, obj)
            elif self.p.loss_function == 'mask_softmax':
                tri_loss = self.mask_softmax(pred, label, obj)
            elif self.p.loss_function == 'mix_loss':
                tri_loss = self.mix_loss(pred, label, self.p.tau_plus, self.p.beta, self.p.estimator, obj)
            elif self.p.loss_function == 'bcel_loss':
                tri_loss = self.bcel_loss(pred, label)
            elif self.p.loss_function == 'magr_loss':
                tri_loss = self.magr_loss(pred, label, obj)

            loss = tri_loss + self.p.subgraph_loss_rate * cl_loss

            if args.change_lr:
                
                torch.cuda.empty_cache()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()
                
                if self.p.lr_scheduler in ['CyclicLR','OneCycleLR']:
                    self.scheduler.step()
                elif self.p.lr_scheduler == 'CosineAnnealingWarmRestarts':
                    self.scheduler.step(epoch - 1 + step / iters)
                
            else:
                loss.backward()
                self.optimizer.step()

            losses.append(loss.item())
            
    
        loss = np.mean(losses)
        return loss


    def isinf_nan_if(self, x):
        if torch.isinf(x).any():
            #raise Exception(f'is inf:{x}')
            return 'inf'
        elif torch.isnan(x).any():
            #raise Exception(f'is nan:{x}')
            return 'nan'
        else:
            return 'success'
        

    def isinf_nan(self, x):
        if torch.isinf(x).any():
            raise Exception(f'is inf:{x}')
            #return 'inf'
        if torch.isnan(x).any():
            raise Exception(f'is nan:{x}')
            #return 'nan'

    def contrastive_loss(self, pred, label, tau_plus, beta, estimator, obj):
        try:
            if self.isinf_nan_if(torch.exp(pred / self.p.temperature)) == 'inf' or self.isinf_nan_if(torch.exp(pred / self.p.temperature)) == 'nan':
                pred = pred / torch.abs(pred).max(dim = -1, keepdim = True)[0]

            if (torch.exp(pred / self.p.temperature) == 0).any():
                pred = torch.exp(pred / self.p.temperature) + 1e-8 
            else:
                pred = torch.exp(pred / self.p.temperature)
            self.isinf_nan(pred)  
            
            loss = 0
            pos = []
            
            for i in range(len(label)):
                if args.lbl_smooth != 0:
                    index_line =  label[i] > 0.9
                else:
                    index_line =  torch.nonzero(label[i]).view(-1)
                if self.p.sample_mod == 'full':  
                    pos_random_idx = obj[i]
                elif self.p.sample_mod == 'part':
                    
                    
                    pos_random_idx = obj[i] 
                else:
                    exit('please gain args.sample_mod')

                rs = pred[i][pos_random_idx].sum()
                pos.append(rs) 

            pos = torch.stack(pos, dim=0)

            neg = []
            index_line_bool = label <= 0
            for i in range(len(label)):
                if args.lbl_smooth != 0: 
                    index_line =  label[i] < 0.1
                else:
                    index_line = torch.nonzero(index_line_bool[i]).view(-1)
                if self.p.sample_mod == 'full':
                    neg_random_idx = index_line 
                elif self.p.sample_mod == 'part':
                    neg_random_idx = random.sample(index_line.cpu().tolist(), self.p.neg_sample_num) 
                else:
                    exit('please gain args.sample_mod')

                rs = pred[i][neg_random_idx].sum()         
                neg.append(rs)

            neg = torch.stack(neg, dim=0)         


            if self.isinf_nan_if((- torch.log(pos / (pos + neg) )).mean()) == 'inf' or self.isinf_nan_if((- torch.log(pos / (pos + neg) )).mean()) == 'nan':
                pos = pos + 1e-8 
                neg = neg + 1e-8        
            loss = (- torch.log(pos / (pos + args.Q * neg) )).mean()  
            self.isinf_nan(loss)
        except Exception as e:
            print(e)
            exit('error')
           
        return loss



    def mask_softmax(self, pred, label, obj):
        loss = 0
        pos = []
        pred_log = torch.log_softmax(pred, dim=-1) / self.p.embed_dim


        for i in range(len(label)):
            index_line = obj[i]
            rs = pred_log[i][index_line].sum()
            pos.append(rs)

        loss = torch.stack(pos, dim=0).mean() 
        return -loss
        

    def bcel_loss(self, pred, label):
        bceloss = torch.nn.BCELoss()
        pred_s = torch.sigmoid(pred)
        return bceloss(pred_s, label) 



    def magr_loss(self, pred, label, obj):
        magrloss= nn.MarginRankingLoss(0.5) 
        loss_list = []
        
        for i in range(len(label)):

            index_line_bool = label <= 0
            neg_random_idx = torch.nonzero(index_line_bool[i]).view(-1)
            n = pred[i][neg_random_idx] 
            pos_random_idx = obj[i] 
            p = pred[i][[pos_random_idx]].repeat(n.size(0)) 
            mg = torch.ones(n.size(0)).cuda()
            loss= magrloss(p, n, mg)
            loss_list.append(loss)
        
        loss = torch.stack(loss_list, dim = 0).mean()
        return loss
    
        
    def mix_loss(self, pred, label, tau_plus, beta, estimator, obj):

        if args.mix_loss_mod == 'CLoss-MLoss':      
            all_loss = self.contrastive_loss(pred, label, tau_plus, beta, estimator, obj) + self.mask_softmax(pred, label, obj)
        elif args.mix_loss_mod == 'CLoss-BCELoss':  
            all_loss = self.contrastive_loss(pred, label, tau_plus, beta, estimator, obj) + self.bcel_loss(pred, label)
        elif args.mix_loss_mod == 'MLoss-BCELoss':  
            all_loss = self.mask_softmax(pred, label, obj) + self.bcel_loss(pred, label)
            
        return all_loss


    

    def fit(self):
        best_val = 0
        best_epoch = 0
        best_lr = 0
        best_mrr = 0
        best_mr = 0
        best_hit1 = 0
        best_hit3 = 0
        best_hit10 = 0
        val_mrr = 0
        self.best_val_mrr, self.best_val, self.best_epoch = 0., {}, 0
        
        save_path = os.path.join(args.model_dir, args.task_name + '_' + args.name)


        kill_cnt = 0
        for epoch in np.arange(self.p.max_epochs) + 1: 
            starting = time.time()      
            train_loss  = self.run_epoch(epoch, val_mrr)
                    
            if args.change_lr and self.p.lr_scheduler not in ['CyclicLR','OneCycleLR','CosineAnnealingWarmRestarts','ReduceLROnPlateau']:
                self.scheduler.step()
            
            val_results, valid_loss = self.evaluate('valid')
            
            if args.change_lr and self.p.lr_scheduler == 'ReduceLROnPlateau':  
                self.scheduler.step(valid_loss)
            
            ending = time.time()
            
            self.logger.info('验证:')
            self.logger.info(("Epoch:{}  time {}m  LR:{:.10f}").format(epoch, ((ending-starting)//60), self.optimizer.param_groups[0]['lr']))
            self.logger.info(("Train Loss:{:.10f}  Valid Loss:{:.10f}").format(train_loss,valid_loss))
            self.logger.info(("Valid MRR:{:.5f}  Valid MR:{:.5f}").format(val_results['mrr'], val_results['mr']))
            self.logger.info(("Valid H@1:{:.5f}  Valid H@3:{:.5f}  Valid H@10:{:.5f}").format(val_results['hits@1'],val_results['hits@3'],val_results['hits@10']))
            self.logger.info('\n\n')

            epoch_save_path = os.path.join(args.model_dir, 'epoch' + str(epoch))
            self.save_model_epoch(epoch_save_path, epoch) 

            if val_results['mrr'] > best_val:
                self.best_val = val_results
                self.best_val_mrr = val_results['mrr']
                self.best_epoch = epoch
                best_val = val_results['mrr']
                best_epoch = epoch
                best_mrr = val_results['mrr']
                best_mr = val_results['mr']
                best_hit1 = val_results['hits@1']
                best_hit3 = val_results['hits@3']
                best_hit10 = val_results['hits@10']
                best_lr = self.optimizer.param_groups[0]['lr']
                self.save_model(save_path)
                kill_cnt = 0

            else:
                kill_cnt += 1
                
                
                if kill_cnt % 10 == 0 and self.p.gamma > 5:
                    self.p.gamma -= 5
                    self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                    self.logger.info('\n\n')
                
                
                
                if args.change_tem == 1:
                    if kill_cnt % 5 == 0 and self.p.save_path != None: 
                        if round(self.p.temperature, 2) >= 1.49:
                            self.p.temperature = 0.1
                        else:
                            self.p.temperature += 0.1
                        self.logger.info('temperature add on saturation, updated value of temperature {}'.format(self.p.temperature))
                        self.logger.info('\n\n')
                        
                    if kill_cnt % 5 == 0 and self.p.save_path == None: 
                        if round(self.p.temperature, 2) <= 0.1: 
                            self.p.temperature = 1.5
                        else:
                            self.p.temperature -= 0.1 
                        self.logger.info('temperature decay on saturation, updated value of temperature {}'.format(self.p.temperature))
                        self.logger.info('\n\n')
                
                    
                if kill_cnt > args.stop_num:
                    self.logger.info("Early Stopping!!")
                    break
      

            if epoch % 5 ==0:
                self.logger.info(f'best_epoch: {best_epoch}')
                self.logger.info(('best_lr: %.10f') % (best_lr))
                self.logger.info(('best_mrr: %.5f') % (best_mrr))
                self.logger.info(('best_mr: %.5f') % (best_mr))
                self.logger.info(('best_hit1: %.5f') % (best_hit1))
                self.logger.info(('best_hit3: %.5f') % (best_hit3))
                self.logger.info(('best_hit10: %.5f') % (best_hit10))
                self.logger.info('\n\n')

        self.logger.info(f'best_epoch: {best_epoch}')
        self.logger.info(('best_lr: %.10f') % (best_lr))
        self.logger.info(('best_mrr: %.5f') % (best_mrr))
        self.logger.info(('best_mr: %.5f') % (best_mr))
        self.logger.info(('best_hit1: %.5f') % (best_hit1))
        self.logger.info(('best_hit3: %.5f') % (best_hit3))
        self.logger.info(('best_hit10: %.5f') % (best_hit10))
        self.logger.info('\n\n')
        
   
        self.load_model(save_path)
        test_results = self.evaluate('test')

        self.logger.info('test:')
        self.logger.info(('Test_mrr: %.5f') % test_results['mrr'])
        self.logger.info(('Test_mr: %.5f') % test_results['mr'])
        self.logger.info(('Test_hit1: %.5f') % test_results['hits@1'])
        self.logger.info(('Test_hit3: %.5f') % test_results['hits@3'])
        self.logger.info(('Test_hit10: %.5f') % test_results['hits@10'])

        localtime = time.asctime(time.localtime(time.time()))
        self.logger.info(f'stop time:{localtime}')
        


    def test(self): 
        self.logger.info('Loading best model, Evaluating on Test data')
        save_path = os.path.join(args.model_dir, args.save_path)
        self.load_model(save_path)
        test_results = self.evaluate('test')

        self.logger.info('\n\n')
        self.logger.info('test:')
        self.logger.info(('Test_mrr: %.5f') % test_results['mrr'])
        self.logger.info(('Test_mr: %.5f') % test_results['mr'])
        self.logger.info(('Test_hit1: %.5f') % test_results['hits@1'])
        self.logger.info(('Test_hit3: %.5f') % test_results['hits@3'])
        self.logger.info(('Test_hit10: %.5f') % test_results['hits@10'])

        localtime = time.asctime(time.localtime(time.time()))
        self.logger.info(f'stop time:{localtime}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name',	    default='nckge',					help='Set run name for saving/restoring models')
    parser.add_argument('--dataset',	dest='dataset',         default='PharmKG8k-28',            help='Dataset to use, default: FB15k-237')
    parser.add_argument('--data_name',  default='fb15k',        type=str,                       help='The data name of preprocessed graph.')
    parser.add_argument('--model',	    dest='model',		    default='rhcgat',		        help='Model Name')
    parser.add_argument('--score_func',	dest='score_func',	    default='conve',		        help='Score Function for Link prediction')
    parser.add_argument('--opn',        dest='opn',             default='corr',                 help='Composition Operation to be used in CompGCN')

    parser.add_argument('--batch_size', dest='batch_size',      default=512,    type=int,       help='Batch size for GAT')
    parser.add_argument('--gamma',	    type=float,             default=40,			            help='Margin, default = 40')
    parser.add_argument('--gpu',	    type=str,               default='0',			        help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--epoch',	    dest='max_epochs', 	    type=int,       default=1000,  	help='Number of epochs')
    parser.add_argument('--l2',		    type=float,             default=0.0,			        help='L2 Regularization for Optimizer')
    parser.add_argument('--lr',		    type=float,             default=0.001,			        help='Starting Learning Rate')
    parser.add_argument('--lbl_smooth', dest='lbl_smooth',	    type=float,     default=0,	    help='Label Smoothing')
    parser.add_argument('--num_workers',type=int,               default=10,                     help='Number of processes to construct batches')
    parser.add_argument('--seed',       dest='seed',            default=41504,  type=int,     	help='Seed for randomization')

    
    parser.add_argument('--bias',       dest='bias',            default=0,      type=int,       help='Whether to use bias in the model')
    
    parser.add_argument('--num_bases',	dest='num_bases', 	    default=-1,   	type=int, 	    help='Number of basis relation vectors to use')
    parser.add_argument('--init_dim',	dest='init_dim',	    default=100,	type=int,	    help='Initial dimension size for entities and relations')
    parser.add_argument('--gcn_dim',	dest='gcn_dim', 	    default=200,   	type=int, 	    help='Number of hidden units in GCN')
    parser.add_argument('--embed_dim',	dest='embed_dim', 	    default=200,    type=int, 	    help='Embedding dimension to give as input to score function')
    parser.add_argument('--gcn_layer',	dest='gcn_layer', 	    default=2,   	type=int, 	    help='Number of GCN Layers to use')
    parser.add_argument('--gcn_drop',	dest='dropout', 	    default=0.1,  	type=float,	    help='Dropout to use in GCN Layer')


    parser.add_argument('--pretrained_emb',         dest='pretrained_emb',      default=0, type=int, help='')
    parser.add_argument('--num_of_layers',          dest='num_of_layers',       default=2, type=int, help='')
    parser.add_argument('--num_heads_per_layer',    dest='num_heads_per_layer', nargs="*", default=[8, 1], type=int, help='')
    parser.add_argument('--add_skip_connection',    dest='add_skip_connection', default=0, type=int, help='')
    parser.add_argument('--bias_gat',               dest='bias_gat',            default=1, type=int, help='')
    parser.add_argument('--dropout_gat',            dest='dropout_gat',         default=0.5, type=float, help='')
    parser.add_argument('--num_features_per_layer', dest='num_features_per_layer', nargs="*",
                        default=[200, 25, 200], type=int, help='') 
    


    
    parser.add_argument('--hid_drop',   dest='hid_drop',    default=0.3,    type=float,     help='ConvE1: Hidden dropout')
    parser.add_argument('--hid_drop2',  dest='hid_drop2',   default=0.4,  	type=float,	    help='ConvE2: Hidden dropout')
    parser.add_argument('--feat_drop', 	dest='feat_drop',   default=0.3,  	type=float,	    help='ConvE: Feature Dropout')
    parser.add_argument('--k_w',	    dest='k_w', 	    default=10,   	type=int, 	    help='ConvE: k_w')
    parser.add_argument('--k_h',	    dest='k_h', 	    default=20,   	type=int, 	    help='ConvE: k_h')
    parser.add_argument('--num_filt',  	dest='num_filt',    default=200,   	type=int, 	    help='ConvE: Number of filters in convolution')
    parser.add_argument('--ker_sz',    	dest='ker_sz', 	    default=8,   	type=int, 	    help='ConvE: Kernel size to use')




    parser.add_argument('--pos_sample_num',       type=int,   default=1,    help='positive sample num')
    
    parser.add_argument('--neg_sample_num',       type=int,   default=100,  help='negtive sample num')


    parser.add_argument('--subgraph_loss_rate',     type=float, default=0.0, help='loss_rate')   

    parser.add_argument('--node_loss_rate',         type=float, default=0.001, help='loss_rate') 

    parser.add_argument('--local_rank',     default=0,  type=int,   help='node rank for distributed training')
    
    parser.add_argument('--use_pretrain',   type=int,   default=0,  help='whether use pretrain')

    parser.add_argument('--use_test',       type=int,   default=0,  help='whether be used in test')

    parser.add_argument('--test',           type=int,   default=0,  help='whether be used in testdata') 

    parser.add_argument('--test_name',      type=str,   default='', help='test name')

    parser.add_argument('--model_dir',      type=str,   default='./fb15k237_model_save', help='The address for storing the models and optimization results.')

    parser.add_argument('--task_name',      type=str,   default='LP', help='The name of the stored models and optimization results.')

    parser.add_argument('--log_dir',         dest='log_dir',     default='./fb15k237_log/',   help='Log directory')

    parser.add_argument('--config',         dest='config_dir',  default='./config/',        help='Config directory')
            
    parser.add_argument('--loss_function',  type=str,   default='contrastive_loss',
                        choices=['contrastive_loss', 'mask_softmax', 'mix_loss','bcel_loss', 'magr_loss'],
                        help='choose loss function')

    parser.add_argument('--mix_loss_mod',  type=str,   default='CLoss-MLoss',
                        choices=['CLoss-MLoss', 'CLoss-BCELoss', 'MLoss-BCELoss'],
                        help='choose loss function')

    parser.add_argument('--sample_mod',     type=str,   default='full',     help='对比学习的全部采样和部分采样')
    
    parser.add_argument('--stop_num',       type=int,   default=50,   help='stop num of epoch')

    parser.add_argument('--save_path',      type=str,   default=None,     help='load model from save path')

    parser.add_argument('--optimizer',      type=str,   default='adam',
                        choices=['adamw', 'adam', 'sgd', 'adagrad'],
                        help='optimizer to use.')
    
    parser.add_argument('--change_lr',      default=0,  type=int,   help='change lr') 

    parser.add_argument('--lr_scheduler',   type=str,   default='CosineAnnealingLR',
                        choices=['CyclicLR', 'OneCycleLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau', 'CosineAnnealingLR'],
                        help='lr scheduler to use.')
    
    parser.add_argument('--clip',           type=float, default=0.25,   help='Gradient Norm Clipping') 

    parser.add_argument('--workdir',        type=str, default='GNN',    help='Gradient Norm Clipping')
    
    parser.add_argument('--conv_name', type=str, default='nckge',
                        choices=['nckge'],
                        help='The name of GNN filter')
    
    parser.add_argument('--conve_act',       type=str, default = 'relu6', choices = ['relu', 'gelu', 'tanh', 'leakyrelu', 'selu', 'hardswish', 'relu6'],
        help='')

    
    
    parser.add_argument('--temperature',    default=1.0,    type=float, help='Temperature used in softmax')
    parser.add_argument('--tau_plus',       default=0.1,    type=float, help='Positive class priorx')
    parser.add_argument('--beta',           default=0.6,    type=float, help='concentration parameter')
    parser.add_argument('--estimator',      default='easy', type=str,   help='Choose loss function')
    parser.add_argument('--Q',              default=1.0,    type=float, help='pso and neg sample of rate') 
    
    parser.add_argument('--change_tem',     default=0,      type=int,   help='whether change tem')
    
    parser.add_argument('--special_test',   default=None,   type=str,   help='whether use special test data')
    parser.add_argument('--store_emb',      default=None,   type=str,   help='save embedding')


    args = parser.parse_args()
    
    
    os.makedirs(args.model_dir, exist_ok = True)
    os.makedirs(args.log_dir, exist_ok = True)

    if args.use_pretrain:
        if args.use_test:
            name = args.test_name + '_' + args.score_func + '_' + args.loss_function + '_' + args.dataset + '_'+ 'CL'+ args.conv_name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H_%M_%S')
        else:
            name = args.score_func + '_' + args.loss_function + '_' + args.dataset + '_'+ 'CL'+ args.conv_name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H_%M_%S')
    else:
        if args.use_test:
            name = args.test_name + '_' + 'notext_' + args.score_func + '_' + args.loss_function + '_' + args.dataset + '_'+ 'CL'+ args.conv_name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H_%M_%S')
        else:
            name = 'notext_' + args.score_func + '_' + args.loss_function + '_' + args.dataset + '_'+ 'CL'+ args.conv_name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H_%M_%S')



    args.name = name

    set_gpu(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.save_path != None:
        args.temperature = args.temperature
    else:
        args.temperature = args.temperature
    

    dist.init_process_group(backend="nccl")
    model = Runner(args)

    if args.test != 1:
        model.fit() 
    else: 
        model.test()
        
    dist.destroy_process_group()
    
    current_directory = os.path.dirname(os.path.abspath('__file__')) 
    pt = os.path.join(current_directory, args.log_dir.replace('./', ''), args.task_name + '_' + args.name + '.log')
    logging.info(('file_path: %s') % (pt)) 
    print('file_path:', pt)
    
    localtime = time.asctime(time.localtime(time.time()))
    logging.info(f'stop time:{localtime}')
    


