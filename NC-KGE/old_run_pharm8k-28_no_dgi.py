from helper import *
from data_loader import *
import sys
# sys.path.append('./')
from model.models_gat_no_dgi import *
from tqdm import tqdm
import os
import time
#import tensorflow as tf
#多关系+对比

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

#单机多卡
local_rank=int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
torch.cuda.set_device(local_rank)

class Runner(object):

    def load_data(self):
        """
        Reading in raw triples and converts it into a standard format.

        Parameters
        ----------
        self.p.dataset:         Takes in the name of the dataset (FB15k-237)

        Returns
        -------
        self.ent2id:            Entity to unique identifier mapping
        self.id2rel:            Inverse mapping of self.ent2id
        self.rel2id:            Relation to unique identifier mapping
        self.num_ent:           Number of entities in the Knowledge graph
        self.num_rel:           Number of relations in the Knowledge graph
        self.embed_dim:         Embedding dimension used
        self.data['train']:     Stores the triples corresponding to training dataset
        self.data['valid']:     Stores the triples corresponding to validation dataset
        self.data['test']:      Stores the triples corresponding to test dataset
        self.data_iter:		The dataloader for different data splits

        """

        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in ['train', 'valid', 'test']:
            for line in open('../data/{}/{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for idx, rel in enumerate(rel_set)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

        self.data = ddict(list)
        sr2o = ddict(set)

        for split in ['train', 'valid', 'test']:
            for line in open('../data/{}/{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, rel, obj))

                #if split == 'train':
                    #sr2o[(sub, rel)].add(obj)
                    #sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.data = dict(self.data)

        #self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in ['train', 'valid', 'test']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = ddict(list)
        
        for split in ['train']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}'.format(split)].append(
                    {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)], 'sub_samp': 1})
                self.triples['{}'.format(split)].append(
                    {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)], 'sub_samp': 1})        


        #for (sub, rel), obj in self.sr2o.items():
            #self.triples['train'].append({'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
            #self.triples['train'].append({'triple': (sub, rel, obj), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

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

        # if self.p.pretrained_emb:
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

    def init_embeddings(entity_file, relation_file):
        entity_emb, relation_emb = [], []

        with open('../data/' + entity_file) as f:
            for line in f:
                entity_emb.append([float(val) for val in line.strip().split()])

        with open('../data/' + relation_file) as f:
            for line in f:
                relation_emb.append([float(val) for val in line.strip().split()])

        return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)

    def construct_adj(self):
        """
        Constructor of the runner class

        Parameters
        ----------

        Returns
        -------
        Constructs the adjacency matrix for GCN

        """
        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        # Adding inverse edges
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)

        edge_index = torch.LongTensor(edge_index).to(self.device).t()
        edge_type = torch.LongTensor(edge_type).to(self.device)

        return edge_index, edge_type

    def __init__(self, params):
        """
        Constructor of the runner class

        Parameters
        ----------
        params:         List of hyper-parameters of the model

        Returns
        -------
        Creates computational graph and optimizer

        """
        self.p = params
       
        self.logger = get_logger(args.task_name + '_'+ args.name + '.log', args.log_dir, args.config_dir)

        
        #self.logger = get_logger(args.task_name + '_' + args.name + '.log', args.workdir, args.config_dir)
        self.logger.setLevel(logging.DEBUG)
        
        
        for arg in vars(args):
            self.logger.info((arg, getattr(args, arg)))#注意使用logger.info打印数据的时候，一定加一个括号，将数据当元组来看
        self.logger.info('\n\n')
        
        

        if self.p.gpu != '-1' and torch.cuda.is_available():
            #self.device = torch.device(f'cuda:{local_rank}')  #self.device
            self.device = torch.device('cuda')  #self.device
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data()  #self.load_model(save_path)
        self.model = self.add_model(self.p.score_func)
            
        
        
        #['CyclicLR', 'OneCycleLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau', 'CosineAnnealingLR']
        if self.p.lr_scheduler == 'CosineAnnealingLR': #效果不错
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = self.p.max_epochs)#余弦退火调整学习率
            #T_max更新学习率的迭代次数


        elif self.p.lr_scheduler == 'ReduceLROnPlateau': #动态衰减学习率 #效果不错
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min')
            #patience更新学习率的迭代次数

        elif self.p.lr_scheduler == 'CyclicLR': #CyclicLR循环学习率,stepsize=（样本个数/batchsize）*（2~10）,要放在batch里面
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.01, max_lr=0.1 )
        #因为cycle_momentum参数默认就是True而Adam优化器本身不带momentum参数的，所以就出现了如题的错误。
        #base_lr, max_lr,效果极差，不再测试，这个2个参数的值，设置大点比较好

        elif self.p.lr_scheduler == 'OneCycleLR': #total_steps:总的batch数,这个参数设置后就不用设置epochs和steps_per_epoch
            self.scheduler =torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=0.01, \
                                                        steps_per_epoch = len(self.data_iter['train']), epochs = self.p.max_epochs)
        #max_lr 设置点比较好

        elif self.p.lr_scheduler == 'CosineAnnealingWarmRestarts': #T_mult = 2试试,效果不如T_mult = 1，暂时不再测试
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0 = 10, T_mult = 1)

        '''
        'CyclicLR','OneCycleLR','CosineAnnealingWarmRestarts'的参数设置是个难点。如果没把握或者不进行大量测试，建议使用上
        面两种方法
        
        '''

    def add_model(self, score_func):
        """
        Creates the computational graph

        Parameters
        ----------
        model_name:     Contains the model name to be created

        Returns
        -------
        Creates the computational graph for model and initializes it

        """
        #model_name = '{}_{}'.format(model, score_func)
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
        else:
            raise NotImplementedError
            
        if self.p.save_path != None:
            print('retrain')
            self.load_model(os.path.join(self.p.model_dir, self.p.save_path))
            
        else:
            print('train')
            self.optimizer = self.add_optimizer(self.model.parameters())
            
        



        #model.to(self.device)
        #多卡
        model = self.model.cuda(local_rank)
        model = DDP(model, [local_rank], find_unused_parameters=True)
        num_param = sum([p.numel() for p in model.parameters()])
        self.logger.info("Number of parameters: {0}".format(num_param)) #统计参数规模吗？可靠吗？怎么换算呢？等于多少M或者KB呢？
        
        return model

    def add_optimizer(self, parameters):
        """
        Creates an optimizer for training the parameters

        Parameters
        ----------
        parameters:         The parameters of the model

        Returns
        -------
        Returns an optimizer for learning the parameters of the model

        """
        if self.p.lr_scheduler != 'CyclicLR': #adamw与adam的唯一区别是，adamw是有默认权重衰减的，其它都一样。
            if args.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(parameters)#AdamW的学习率会自动变化，学习率默认: 1e-3，权重衰减默认: 1e-2
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)
                #optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.99), eps=1e-05, weight_decay=args.l2) #不如默认的adam配置
                #Adam的学习率，如果weight_decay为0，则学习率不改变。默认为0。如果为0，则我们使用的各种学习学习率调整策略就无意义的
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(parameters, lr=0.001)
            elif args.optimizer == 'adagrad':
                optimizer = torch.optim.Adagrad(parameters)
        else:
            optimizer = torch.optim.SGD(parameters, lr=0.001)
            
        return optimizer

    def read_batch(self, batch, split):
        """
        Function to read a batch of data and move the tensors in batch to CPU/GPU

        Parameters
        ----------
        batch: 		the batch to process
        split: (string) If split == 'train', 'valid' or 'test' split


        Returns
        -------
        Head, Relation, Tails, labels
        """
        if split == 'train':
            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label
        else:
            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label

    def save_model(self, save_path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

        Parameters
        ----------
        save_path: path where the model is saved

        Returns
        -------
        """
        state = {
            'state_dict': self.model.state_dict(),
            'best_val': self.best_val,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args'	: vars(self.p)
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        """
        Function to load a saved model

        Parameters
        ----------
        load_path: path to the saved model

        Returns
        -------
        """
        state			= torch.load(load_path)
        state_dict		= state['state_dict']
        self.best_val		= state['best_val']
        self.best_val_mrr	= self.best_val['mrr']

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def evaluate(self, split):
        """
        Function to evaluate the model on validation or test set

        Parameters
        ----------
        split: (string) If split == 'valid' then evaluate on the validation set, else the test set
        epoch: (int) Current epoch count

        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        if split == 'valid':
            left_results,left_loss = self.predict(split=split, mode='tail_batch')
            right_results,right_loss = self.predict(split=split, mode='head_batch')
            valid_loss = (left_loss + right_loss) / 2
            results = get_combined_results(left_results, right_results)
            #logger.info(
                #'[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'],
                                                                                     #results['right_mrr'], results['mrr']))
            return results,valid_loss
        else:
            left_results = self.predict(split=split, mode='tail_batch')
            right_results = self.predict(split=split, mode='head_batch')
            results = get_combined_results(left_results, right_results)
            return results

    def predict(self, split='valid', mode='tail_batch'):
        """
        Function to run model evaluation for a given mode

        Parameters
        ----------
        split: (string) 	If split == 'valid' then evaluate on the validation set, else the test set
        mode: (string):		Can be 'head_batch' or 'tail_batch'

        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        self.model.eval()

        with torch.no_grad():
            results = {}
            losses = []
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

            for step, batch in enumerate(train_iter):

                if split == 'valid':
                    sub, rel, obj, label = self.read_batch(batch, 'valid')
                    pred, cl_loss = self.model.forward(sub, rel)
                    #tri_loss = self.model.loss(pred, label)

                    if self.p.loss_function == 'contrastive_loss':
                        tri_loss = self.contrastive_loss(pred, label, self.p.tau_plus, self.p.beta, self.p.estimator, obj)
                    elif self.p.loss_function == 'mask_softmax':
                        tri_loss = self.mask_softmax(pred, label, obj)
                    elif self.p.loss_function == 'mix_loss':
                        tri_loss = self.mix_loss(pred, label, self.p.tau_plus, self.p.beta, self.p.estimator, obj)
                    elif self.p.loss_function == 'bcel_loss':
                        tri_loss = self.bcel_loss(pred, label)#多标签分类任务，所以label是矩阵的形式，而非向量，不同于二元分类和多元分类时传递真值的下标向量

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

                #if step % 100 == 0:
                    #logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))
                    #title()首字母大写,这里的输出没意义
            if split == 'valid':
                valid_loss = np.mean(losses)
                return results,valid_loss
            else:
                return results










    def run_epoch(self, epoch, val_mrr = 0):
        """
        Function to run one epoch of training

        Parameters
        ----------
        epoch: current epoch count

        Returns
        -------
        loss: The loss value after the completion of one epoch
        """
        #change_lr = True #是否动态更改学习率
        self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train'])
        iters = len(self.data_iter['train'])
        b_xent = nn.BCEWithLogitsLoss() #未使用
        lbl = torch.cat((torch.ones(1, self.p.num_ent), torch.zeros(1, self.p.num_ent)), 1).cuda() #未使用
        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            sub, rel, obj, label = self.read_batch(batch, 'train')
            pred, cl_loss = self.model.forward(sub, rel)
            #tri_loss = self.model.loss(pred, label)

            if self.p.loss_function == 'contrastive_loss':
                tri_loss = self.contrastive_loss(pred, label, self.p.tau_plus, self.p.beta, self.p.estimator, obj)
            elif self.p.loss_function == 'mask_softmax':
                tri_loss = self.mask_softmax(pred, label, obj)
            elif self.p.loss_function == 'mix_loss':
                tri_loss = self.mix_loss(pred, label, self.p.tau_plus, self.p.beta, self.p.estimator, obj)
            elif self.p.loss_function == 'bcel_loss':
                tri_loss = self.bcel_loss(pred, label)#多标签分类任务，所以label是矩阵的形式，而非向量，不同于二元分类和多元分类时传递真值的下标向量

            loss = tri_loss + self.p.subgraph_loss_rate * cl_loss

            if args.change_lr:
                #新的优化器，学习率自动变化
                #self.optimizer.zero_grad() 
                torch.cuda.empty_cache()
                #loss.backward(retain_graph=True)
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




    def run_epoch_old(self, epoch, val_mrr=0):#未使用
        """
        Function to run one epoch of training

        Parameters
        ----------
        epoch: current epoch count

        Returns
        -------
        loss: The loss value after the completion of one epoch
        """
        self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train'])
        iters = len(self.data_iter['train'])#获取批量数量，控制学习率的步长变化
        
        for step, batch in enumerate(train_iter):
            #print('trian')
            self.optimizer.zero_grad()
            sub, rel, obj, label = self.read_batch(batch, 'train')
            x, r, y = self.gnn.forward(sub, rel)
            pred = self.matcher.forward(x, r, y)

            if self.p.loss_function == 'contrastive_loss':
                loss = self.contrastive_loss(pred, label, self.p.tau_plus, self.p.beta, self.p.estimator, obj)
            elif self.p.loss_function == 'mask_softmax':
                loss = self.mask_softmax(pred, label, obj)
            elif self.p.loss_function == 'mix_loss':
                loss = self.mix_loss(pred, label, self.p.tau_plus, self.p.beta, self.p.estimator, obj)
            elif self.p.loss_function == 'bcel_loss':
                loss = self.bcel_loss(pred, label)#多标签分类任务，所以label是矩阵的形式，而非向量，不同于二元分类和多元分类时传递真值的下标向量

            if args.change_lr:
                #新的优化器，学习率自动变化
                torch.cuda.empty_cache()
                #loss.backward(retain_graph=True)
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





    def isinf_nan(self, x):
        if torch.isinf(x).any():
            raise Exception(f'is inf:{x}')
        
        if torch.isnan(x).any():
            raise Exception(f'is nan:{x}')
        
        


    def contrastive_loss_old(self, pred, label, tau_plus, beta, estimator):
        st = time.time()
        '''
            使用对比损失，考虑一下温度系数的动态调整，随着训练次数的增加而逐渐扩
        print('pred.shape:', pred.shape)
        print('label.shape:', label.shape)
        print('label[3]:',label[3])
        print('index_line:',index_line) #这里返回的是所有下标，显然label的取值不是01而是使用了标签平滑,--lbl_smooth == 0
        print('pred[3]:',pred[3])
        '''
        #pred = pred / self.p.embed_dim   #对比学习需要缩放，否则容易溢出，出现nan
        #pred = torch.softmax(pred, dim = -1) #不要用softmax
        self.isinf_nan(pred)
        self.isinf_nan(pred / self.p.temperature)
        #向量单位化,相当于缩放
        #pred_norm = torch.norm(pred, p=2, dim=-1, keepdim=True)
        #pred = pred / pred_norm

        #每行除以最大值 缩放不行
        #pred_norm = torch.max(pred, dim=-1, keepdim=True)[0]

        #BN归一化方法
        
        pred = torch.exp(pred / self.p.temperature)
        #pred = pred / self.p.temperature
        #要判断输出是否含有inf或nan
        self.isinf_nan(pred)  #这一步出现nan
        
        loss = 0
        pos = []
        
        #print('label:', label[:3])
        #print('pos:', label[:3] > 0.9)
        #print('neg:', label[:3] < 0.1)

        for i in range(len(label)):
            if args.lbl_smooth != 0:
                index_line =  label[i] > 0.9
            else:
                index_line =  torch.nonzero(label[i]).view(-1)#找每一行的非零元素下标
            #pos_random_idx = random.sample(index_line.cpu().tolist(), 1)
            #pos_random_idx = index_line[0:1]
            if self.p.sample_mod == 'full':
                pos_random_idx = index_line
            elif self.p.sample_mod == 'part':
                #pos_random_idx = index_line[0:1]
                pos_random_idx = random.sample(index_line.cpu().tolist(), self.p.pos_sample_num)
            else:
                exit('please gain args.sample_mod')

            rs = pred[i][pos_random_idx].sum()#对每一行加和。根据经验，加法运算不合适，因为不同的样本，正样本的数量不同，如果使用加法则导致损失忽高忽低，难以有效指导模型
            #建议改成均值
            #pos.append(rs.view(-1)) #shape = [1]
            pos.append(rs) #shape = [1]

        pos = torch.stack(pos, dim=0)#注意使用stack处理有梯度的张量列表时不会丢失梯度，其他链接方法，如直接将列表转化成张量，会丢失梯度的shape = [B,1] 

        
        if estimator=='hard':
                N = LL-1
                imp = (beta* neg.log()).exp()
                reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
                Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
                # constrain (optional)
                Ng = torch.clamp(Ng, min = N * np.e**(-1 / args.temperature))
                
        elif estimator == 'easy':
            neg = []
            index_line_bool = label <= 0
            for i in range(len(label)):
                if args.lbl_smooth != 0: #如果标签平滑，则原来的0，现在 < 1，原来的1，现在 > 2
                    index_line =  label[i] < 0.1
                else:
                    index_line = torch.nonzero(index_line_bool[i]).view(-1)
                #neg_random_idx = random.sample(index_line.cpu().tolist(), 100) #减少负样本的数量，可以避免溢出
                #neg_random_idx = index_line[0:100]
                if self.p.sample_mod == 'full':
                    neg_random_idx = index_line
                elif self.p.sample_mod == 'part':
                    #neg_random_idx = index_line[0:100]
                    neg_random_idx = random.sample(index_line.cpu().tolist(), self.p.neg_sample_num)
                else:
                    exit('please gain args.sample_mod')

                #print(neg_random_idx)
                #print(pred[i][neg_random_idx].shape) #torch.Size([200])
                #exit()
                rs = pred[i][neg_random_idx].sum()
                neg.append(rs)

            
            neg = torch.stack(neg, dim=0)         
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

        #print('label:', label)
        #print('pos:', pos)
        #print('neg:', neg)
        #exit()

        #要判断输出是否含有inf或nan
        self.isinf_nan(pos)
        #要判断输出是否含有inf或nan
        self.isinf_nan(neg)
        self.isinf_nan(pos / (pos + neg))
        self.isinf_nan(torch.log(pos / (pos + neg)))   #出现溢出   [torch.tensor(10e-30), torch.tensor(10e30)]这个范围之外会溢出   
        self.isinf_nan(- torch.log(pos / (pos + neg)))  #加减乘除是无法检测出Inf的，而troch.exp(), torch.log()操作则会溢出，报nan错误
        
        loss = (- torch.log(pos / (pos + neg) )).mean()  #pos>All导致loss为负
        #loss = (- torch.log(pos / (pos + neg) )).sum()  #pos>All导致loss为负
        self.isinf_nan(loss)
        
        en = time.time()
        #print('CL time:{}S'.format(en - st))
            
        return loss





    def contrastive_loss(self, pred, label, tau_plus, beta, estimator, obj):
        st = time.time()
        #print('obj: {}'.format(obj))
        #print('len(obj): {}'.format(len(obj)))
        
        '''
            使用对比损失，考虑一下温度系数的动态调整，随着训练次数的增加而逐渐扩
        print('pred.shape:', pred.shape)
        print('label.shape:', label.shape)
        print('label[3]:',label[3])
        print('index_line:',index_line) #这里返回的是所有下标，显然label的取值不是01而是使用了标签平滑,--lbl_smooth == 0
        print('pred[3]:',pred[3])
        '''
        #pred = pred / self.p.embed_dim   #对比学习需要缩放，否则容易溢出，出现nan
        #pred = torch.softmax(pred, dim = -1) #不要用softmax
        self.isinf_nan(pred)
        self.isinf_nan(pred / self.p.temperature)
        #向量单位化,相当于缩放
        #pred_norm = torch.norm(pred, p=2, dim=-1, keepdim=True)
        #pred = pred / pred_norm

        #每行除以最大值 缩放不行
        #pred_norm = torch.max(pred, dim=-1, keepdim=True)[0]

        #BN归一化方法
        
        pred = torch.exp(pred / self.p.temperature)
        #pred = pred / self.p.temperature
        #要判断输出是否含有inf或nan
        self.isinf_nan(pred)  #这一步出现nan
        
        loss = 0
        pos = []
        
        #print('label:', label[:3])
        #print('pos:', label[:3] > 0.9)
        #print('neg:', label[:3] < 0.1)

        for i in range(len(label)):
            if args.lbl_smooth != 0:
                index_line =  label[i] > 0.9
            else:
                index_line =  torch.nonzero(label[i]).view(-1)#找每一行的非零元素下标
            #pos_random_idx = random.sample(index_line.cpu().tolist(), 1)
            #pos_random_idx = index_line[0:1]
            if self.p.sample_mod == 'full':
                pos_random_idx = index_line
            elif self.p.sample_mod == 'part':
                #pos_random_idx = index_line[0:1]
                #pos_random_idx = random.sample(index_line.cpu().tolist(), self.p.pos_sample_num)
                pos_random_idx = obj[i] #obj存放的是每一个三元组的真实尾节点id，这里因为只取一个正样本，所以该样本来自于obj
            else:
                exit('please gain args.sample_mod')

            rs = pred[i][pos_random_idx].sum()#对每一行加和。根据经验，加法运算不合适，因为不同的样本，正样本的数量不同，如果使用加法则导致损失忽高忽低，难以有效指导模型
            #建议改成均值
            #pos.append(rs.view(-1)) #shape = [1]
            pos.append(rs) #shape = [1]

        pos = torch.stack(pos, dim=0)#注意使用stack处理有梯度的张量列表时不会丢失梯度，其他链接方法，如直接将列表转化成张量，会丢失梯度的shape = [B,1] 

        
        if estimator=='hard':
                N = LL-1
                imp = (beta* neg.log()).exp()
                reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
                Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
                # constrain (optional)
                Ng = torch.clamp(Ng, min = N * np.e**(-1 / args.temperature))
                
        elif estimator == 'easy':
            neg = []
            index_line_bool = label <= 0
            for i in range(len(label)):
                if args.lbl_smooth != 0: #如果标签平滑，则原来的0，现在 < 1，原来的1，现在 > 2
                    index_line =  label[i] < 0.1
                else:
                    index_line = torch.nonzero(index_line_bool[i]).view(-1)
                #neg_random_idx = random.sample(index_line.cpu().tolist(), 100) #减少负样本的数量，可以避免溢出
                #neg_random_idx = index_line[0:100]
                if self.p.sample_mod == 'full':
                    neg_random_idx = index_line
                elif self.p.sample_mod == 'part':
                    #neg_random_idx = index_line[0:100]
                    neg_random_idx = random.sample(index_line.cpu().tolist(), self.p.neg_sample_num)
                else:
                    exit('please gain args.sample_mod')

                #print(neg_random_idx)
                #print(pred[i][neg_random_idx].shape) #torch.Size([200])
                #exit()
                rs = pred[i][neg_random_idx].sum()
                neg.append(rs)

            
            neg = torch.stack(neg, dim=0)         
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

        #print('label:', label)
        #print('pos:', pos)
        #print('neg:', neg)
        #exit()

        #要判断输出是否含有inf或nan
        self.isinf_nan(pos)
        #要判断输出是否含有inf或nan
        self.isinf_nan(neg)
        self.isinf_nan(pos / (pos + neg))
        self.isinf_nan(torch.log(pos / (pos + neg)))   #出现溢出   [torch.tensor(10e-30), torch.tensor(10e30)]这个范围之外会溢出   
        self.isinf_nan(- torch.log(pos / (pos + neg)))  #加减乘除是无法检测出Inf的，而troch.exp(), torch.log()操作则会溢出，报nan错误
        
        loss = (- torch.log(pos / (pos + neg) )).mean()  #pos>All导致loss为负
        #loss = (- torch.log(pos / (pos + neg) )).sum()  #pos>All导致loss为负
        self.isinf_nan(loss)
        
        en = time.time()
        #print('CL time:{}S'.format(en - st))
            
        return loss



    def mask_softmax_old(self, pred, label):
        #建议这里使用损失的均值，要不然初始损失有点大，影响mix_loss
        loss = 0
        pos = []
        #pred = pred / self.p.embed_dim
        pred_log = torch.log_softmax(pred, dim=-1) / self.p.embed_dim
        #pred_log = torch.log_softmax(pred, dim=-1)
        #softmax对缩放敏感，sigmoid也敏感，因为值缩小之后趋近与0，经过sigmoid的输出都接近于0.5， 而softmax的输出也接近一样于0.16，所以不能乱缩放
        #对pred求softmax,然后取每一行的真值，并相加，实际上是最大化正样本得分损失

        for i in range(len(label)):
            index_line =  torch.nonzero(label[i]).view(-1)#找每一行的非零元素下标
            rs = pred_log[i][index_line].sum()#对每一行加和
            #rs = pred_log[i][index_line].mean()#对每一行求均值， 效果不如求和.一般情况下，同样的效果下，损失大一点效果更好
            pos.append(rs)

        loss = torch.stack(pos, dim=0).mean() #两者差异很小
        #loss = torch.stack(pos, dim=0).sum()
        return -loss
        


    def mask_softmax(self, pred, label, obj):
        #建议这里使用损失的均值，要不然初始损失有点大，影响mix_loss
        loss = 0
        pos = []
        #pred = pred / self.p.embed_dim
        pred_log = torch.log_softmax(pred, dim=-1) / self.p.embed_dim
        #pred_log = torch.log_softmax(pred, dim=-1)
        #softmax对缩放敏感，sigmoid也敏感，因为值缩小之后趋近与0，经过sigmoid的输出都接近于0.5， 而softmax的输出也接近一样于0.16，所以不能乱缩放
        #对pred求softmax,然后取每一行的真值，并相加，实际上是最大化正样本得分损失

        for i in range(len(label)):
            #index_line =  torch.nonzero(label[i]).view(-1)#找每一行的非零元素下标
            index_line = obj[i]
            rs = pred_log[i][index_line].sum()#对每一行加和
            #rs = pred_log[i][index_line].mean()#对每一行求均值， 效果不如求和.一般情况下，同样的效果下，损失大一点效果更好
            pos.append(rs)

        loss = torch.stack(pos, dim=0).mean() #两者差异很小
        #loss = torch.stack(pos, dim=0).sum()
        return -loss
        

    def bcel_loss(self, pred, label):
        bceloss = torch.nn.BCELoss()#多标签分类
        pred_s = torch.sigmoid(pred)
        return bceloss(pred_s, label) 


    def mix_loss(self, pred, label, tau_plus, beta, estimator, obj):

        '''
        将2种损失加在一起看看效果',结果是nan,梯度消失.现在有个问题：对比损失+多标签分类损失有问题，报错。但是单独跑是没问题的,
        出错的原因是对比损失为nan导致无法加法运算
        如果使用的混合损失，则需要两组预测值，显然，我们目前只有一组预测值，且没有经过sigmod，因此不适合BCELoss

        '''
        if args.mix_loss_mod == 'CLoss-MLoss':      #能运行
            all_loss = self.contrastive_loss(pred, label, tau_plus, beta, estimator, obj) + self.mask_softmax(pred, label, obj)
        elif args.mix_loss_mod == 'CLoss-BCELoss':  #能运行
            all_loss = self.contrastive_loss(pred, label, tau_plus, beta, estimator, obj) + self.bcel_loss(pred, label)
        elif args.mix_loss_mod == 'MLoss-BCELoss':  #能运行
            all_loss = self.mask_softmax(pred, label, obj) + self.bcel_loss(pred, label)
            
        #all_loss = self.p.loss_rate * self.contrastive_loss(pred, label, tau_plus, beta, estimator) + (1 - self.p.loss_rate) * self.bcel_loss(pred, label)
        #all_loss = self.p.loss_rate * contrastive_loss(pred, label, tau_plus, beta, estimator) + mask_softmax(pred, label)
        return all_loss




    
    def test(self): #测试集可以单独拿来用，不用再调用fit()
        self.logger.info('Loading best model, Evaluating on Test data')
        save_path = os.path.join(args.model_dir, args.save_path)
        self.load_model(save_path)
        test_results = self.evaluate('test')

        self.logger.info('\n\n')
        self.logger.info('测试:')
        self.logger.info(('Test_mrr: %.5f') % test_results['mrr'])
        self.logger.info(('Test_mr: %.5f') % test_results['mr'])
        self.logger.info(('Test_hit1: %.5f') % test_results['hits@1'])
        self.logger.info(('Test_hit3: %.5f') % test_results['hits@3'])
        self.logger.info(('Test_hit10: %.5f') % test_results['hits@10'])

        localtime = time.asctime(time.localtime(time.time()))
        self.logger.info(f'程序终止时间:{localtime}')
        

    def fit(self):
        """
        Function to run training and evaluation of model

        Parameters
        ----------

        Returns
        -------
        """
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
        
        save_path = os.path.join(args.model_dir, args.task_name + '_' + args.name)#这个作为新模型的名字
        '''
        if self.p.restore:
            self.load_model(os.path.join(args.model_dir, args.save_path)) #如果是加载模型，则只需要传递一个新args.save_path

            self.logger.info('Successfully Loaded previous model')
        '''


        kill_cnt = 0
        for epoch in np.arange(self.p.max_epochs) + 1: 
            starting = time.time()      
            train_loss  = self.run_epoch(epoch, val_mrr)
                    
            if args.change_lr and self.p.lr_scheduler not in ['CyclicLR','OneCycleLR','CosineAnnealingWarmRestarts','ReduceLROnPlateau']:
                self.scheduler.step()
            
            val_results, valid_loss = self.evaluate('valid')
            
            if args.change_lr and self.p.lr_scheduler == 'ReduceLROnPlateau':  #需要在验证损失之后
                self.scheduler.step(valid_loss)
            
            ending = time.time()
            
            self.logger.info('验证:')
            self.logger.info(("Epoch:{}  time {}m  LR:{:.10f}").format(epoch, ((ending-starting)//60), self.optimizer.param_groups[0]['lr']))
            self.logger.info(("Train Loss:{:.10f}  Valid Loss:{:.10f}").format(train_loss,valid_loss))
            self.logger.info(("Valid MRR:{:.5f}  Valid MR:{:.5f}").format(val_results['mrr'], val_results['mr']))
            self.logger.info(("Valid H@1:{:.5f}  Valid H@3:{:.5f}  Valid H@10:{:.5f}").format(val_results['hits@1'],val_results['hits@3'],val_results['hits@10']))
            self.logger.info('\n\n')
            
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
                #torch.save(self.model, os.path.join(args.model_dir, args.task_name + '_' + args.name))#保存模型
                kill_cnt = 0

            else:
                kill_cnt += 1
                if kill_cnt % 10 == 0 and self.p.gamma > 5:
                    self.p.gamma -= 5
                    self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
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
        
        #self.logger.info('Loading best model, Evaluating on Test data')
        self.load_model(save_path)
        #torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.name))#加载模型
        test_results = self.evaluate('test')

        #self.logger.info('\n\n')
        self.logger.info('测试:')
        self.logger.info(('Test_mrr: %.5f') % test_results['mrr'])
        self.logger.info(('Test_mr: %.5f') % test_results['mr'])
        self.logger.info(('Test_hit1: %.5f') % test_results['hits@1'])
        self.logger.info(('Test_hit3: %.5f') % test_results['hits@3'])
        self.logger.info(('Test_hit10: %.5f') % test_results['hits@10'])

        localtime = time.asctime(time.localtime(time.time()))
        self.logger.info(f'程序终止时间:{localtime}')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name',	    default='rhcgat',					help='Set run name for saving/restoring models')
    parser.add_argument('--dataset',	dest='dataset',         default='New_PharmKG8k-28',            help='Dataset to use, default: FB15k-237')
    parser.add_argument('--model',	    dest='model',		    default='rhcgat',		        help='Model Name')
    parser.add_argument('--score_func',	dest='score_func',	    default='conve',		        help='Score Function for Link prediction')
    parser.add_argument('--opn',        dest='opn',             default='corr',                 help='Composition Operation to be used in CompGCN')

    parser.add_argument('--batch_size', dest='batch_size',      default=256,    type=int,       help='Batch size for GAT')
    parser.add_argument('--gamma',	    type=float,             default=40.0,			        help='Margin')
    parser.add_argument('--gpu',	    type=str,               default='0',			        help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--epoch',	    dest='max_epochs', 	    type=int,       default=1000,  	help='Number of epochs')
    parser.add_argument('--l2',		    type=float,             default=0.0,			        help='L2 Regularization for Optimizer')
    parser.add_argument('--lr',		    type=float,             default=0.001,			        help='Starting Learning Rate')
    parser.add_argument('--lbl_smooth', dest='lbl_smooth',	    type=float,     default=0,	    help='Label Smoothing') #默认0.1，使用对比损失，必须设置为0
    parser.add_argument('--num_workers',type=int,               default=10,                     help='Number of processes to construct batches')
    parser.add_argument('--seed',       dest='seed',            default=41504,  type=int,     	help='Seed for randomization')

    #parser.add_argument('--restore',   dest='restore',         action='store_true',            help='Restore from the previously saved model')
    parser.add_argument('--bias',       dest='bias',            default=0,      type=int,       help='Whether to use bias in the model')
    #parser.add_argument('--hid_drop',   dest='hid_drop',        default=0.3, type=float, help='Dropout after GCN')
    parser.add_argument('--num_bases',	dest='num_bases', 	    default=-1,   	type=int, 	    help='Number of basis relation vectors to use')
    parser.add_argument('--init_dim',	dest='init_dim',	    default=100,	type=int,	    help='Initial dimension size for entities and relations')
    parser.add_argument('--gcn_dim',	dest='gcn_dim', 	    default=200,   	type=int, 	    help='Number of hidden units in GCN')
    parser.add_argument('--embed_dim',	dest='embed_dim', 	    default=200,    type=int, 	    help='Embedding dimension to give as input to score function')
    parser.add_argument('--gcn_layer',	dest='gcn_layer', 	    default=2,   	type=int, 	    help='Number of GCN Layers to use')
    parser.add_argument('--gcn_drop',	dest='dropout', 	    default=0.1,  	type=float,	    help='Dropout to use in GCN Layer')


    # arguments for GAT.不要使用bool类型了，作用和action='store_true',一样，如果指定值，则为真，否则使用默认值。使用0与1最好
    parser.add_argument('--pretrained_emb',         dest='pretrained_emb',      default=0, type=int, help='')
    parser.add_argument('--num_of_layers',          dest='num_of_layers',       default=2, type=int, help='')
    parser.add_argument('--num_heads_per_layer',    dest='num_heads_per_layer', nargs="*", default=[8, 1], type=int, help='')#两层网络，每一层的注意力头数量
    parser.add_argument('--add_skip_connection',    dest='add_skip_connection', default=0, type=int, help='')
    parser.add_argument('--bias_gat',               dest='bias_gat',            default=1, type=int, help='')
    parser.add_argument('--dropout_gat',            dest='dropout_gat',         default=0.5, type=float, help='')
    parser.add_argument('--num_features_per_layer', dest='num_features_per_layer', nargs="*",
                        default=[200, 25, 200], type=int, help='') #25是每一个注意力头的维度，即200/8 = 25
    
    #改一下注意力头和网络层的执行，要更灵活，而不是只能执行2层网络。

    # ConvE specific hyperparameters
    parser.add_argument('--hid_drop',   dest='hid_drop',    default=0.3,    type=float,     help='ConvE1: Hidden dropout')
    parser.add_argument('--hid_drop2',  dest='hid_drop2',   default=0.4,  	type=float,	    help='ConvE2: Hidden dropout')
    parser.add_argument('--feat_drop', 	dest='feat_drop',   default=0.3,  	type=float,	    help='ConvE: Feature Dropout')
    parser.add_argument('--k_w',	    dest='k_w', 	    default=10,   	type=int, 	    help='ConvE: k_w')
    parser.add_argument('--k_h',	    dest='k_h', 	    default=20,   	type=int, 	    help='ConvE: k_h')
    parser.add_argument('--num_filt',  	dest='num_filt',    default=200,   	type=int, 	    help='ConvE: Number of filters in convolution')
    parser.add_argument('--ker_sz',    	dest='ker_sz', 	    default=8,   	type=int, 	    help='ConvE: Kernel size to use')

    #parser.add_argument('--logdir',          dest='log_dir',         default='./log/',               help='Log directory')
    #parser.add_argument('--config',          dest='config_dir',      default='./config/',            help='Config directory')


    #一些新参数
    parser.add_argument('--pos_sample_num',       type=int,   default=1,    help='positive sample num')
    
    parser.add_argument('--neg_sample_num',       type=int,   default=100,  help='negtive sample num')

    #设置2种损失函数函数的比重
    parser.add_argument('--subgraph_loss_rate',     type=float, default=0.0, help='loss_rate')   #默认0.001

    parser.add_argument('--node_loss_rate',         type=float, default=0.001, help='loss_rate') #目前没用

    parser.add_argument('--local_rank',     default=0,  type=int,   help='node rank for distributed training')
    
    parser.add_argument('--use_pretrain',   type=int,   default=0,  help='whether use pretrain')

    parser.add_argument('--use_test',       type=int,   default=0,  help='whether be used in test')

    parser.add_argument('--test',           type=int,   default=0,  help='whether be used in testdata') #配合save_path使用，仅用于测试数据集

    parser.add_argument('--test_name',      type=str,   default='', help='test name')

    parser.add_argument('--model_dir',      type=str,   default='./pharm8k_model_save', help='The address for storing the models and optimization results.')

    parser.add_argument('--task_name',      type=str,   default='LP', help='The name of the stored models and optimization results.')

    parser.add_argument('--logdir',         dest='log_dir',     default='./pharm8k_log/',   help='Log directory')

    parser.add_argument('--config',         dest='config_dir',  default='./config/',        help='Config directory')
            
    parser.add_argument('--loss_function',  type=str,   default='bcel_loss',
                        choices=['contrastive_loss', 'mask_softmax', 'mix_loss','bcel_loss'],
                        help='choose loss function')#改一下,多标签损失

    parser.add_argument('--mix_loss_mod',  type=str,   default='CLoss-MLoss',
                        choices=['CLoss-MLoss', 'CLoss-BCELoss', 'MLoss-BCELoss'],
                        help='choose loss function')#改一下,多标签损失

    parser.add_argument('--sample_mod',     type=str,   default='part',     help='对比学习的全部采样和部分采样')
    
    parser.add_argument('--stop_num',       type=int,   default=100,   help='stop num of epoch')

    parser.add_argument('--save_path',      type=str,   default=None,     help='load model from save path')

    parser.add_argument('--optimizer',      type=str,   default='adam',
                        choices=['adamw', 'adam', 'sgd', 'adagrad'],#学习率的优化，注意所给的学习率优化器必须在choice里面
                        help='optimizer to use.')
    
    parser.add_argument('--change_lr',      default=0,  type=int,   help='change lr') #优先使用动态调整学习率的方法，找到最优之后再使用静态的学习率

    parser.add_argument('--lr_scheduler',   type=str,   default='CosineAnnealingLR',
                        choices=['CyclicLR', 'OneCycleLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau', 'CosineAnnealingLR'],#学习率的优化，注意所给的学习率优化器必须在choice里面
                        help='lr scheduler to use.')
    
    parser.add_argument('--clip',           type=float, default=0.25,   help='Gradient Norm Clipping') #梯度修剪，防止梯度爆炸吗？是的

    parser.add_argument('--workdir',        type=str, default='GNN',    help='Gradient Norm Clipping')
    
    parser.add_argument('--conv_name', type=str, default='cmgat',#根据不同的模型名，执行不同的模型,这个看看如何切换
                        choices=['rhcgat', 'rhcgcn', 'rahcgt', 'transe', 'distmult', 'complex', 'simple',\
                                 'origin_sage', 'lgtnv3', 'lgtnv2', 'lgtn', 'rgcnv3', 'rgcnv2', 'rel_dense_mthgcl2', 'gatv4',\
                                 'gatv3', 'hrgat', 'heat', 'eg', 'film', 'gcnv2', 'gatv2', 'transformer', 'fa', 'supergat',\
                                 'indrel_dense_mthgcl', 'rel_dense_mthgcl', 'norel_dense_mthgcl', 'mthgcl', 'dense_mthgcl',\
                                 'mtmrhgnn', 'dense_mtmrhgnn', 'hgcl', 'dense_hgcl', 'compgcn', 'sage', 'hgt', 'gcn', 'gat', 'rgcn',\
                                 'han', 'hetgnn', 'sagecn'],
                        help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
    
    '''
    # python run.py -name best_model -score_func conve -opn corr #注意字符串后面加一个空格。这里的优先级高于命令行参数
    
    cmd = "--dataset PharmKG8k-28 " \
          "--name rhcgat " \
          "--score_func conve " \
          "--opn corr "\
          "--epoch 500 "\
          "--optimizer adam "\
          
    sys.argv += cmd.split()#自动识别为命令行参数
    
    '''


    '''
    HCL的对比损失参数

    '''
    parser.add_argument('--temperature',    default=0.2,    type=float, help='Temperature used in softmax')
    parser.add_argument('--tau_plus',       default=0.1,    type=float, help='Positive class priorx')
    parser.add_argument('--beta',           default=0.6,    type=float, help='concentration parameter')
    parser.add_argument('--estimator',      default='easy', type=str,   help='Choose loss function')

    args = parser.parse_args()
    
    #logger = get_logger( 'stdout.log', args.workdir, args.config_dir) #新方法为什么依旧没有标准输出呀？
    #logger.setLevel(logging.DEBUG)
    

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

    #args.name = name#新增参数
    

    args.name = name#新增参数

    set_gpu(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    '''
    args.file_name = file_name         = time.strftime('%d_%m_%Y') + '_' + time.strftime('%H_%M_%S') 
    if args.use_test:
        #args.workdir  = 'workdir/' + args.test_name + '_' + file_name 
        args.workdir  = 'workdir/' + args.task_name + '_'+ args.name
    else:
        #args.workdir  = 'workdir/' + file_name 
        args.workdir  = 'workdir/' + args.task_name + '_'+ args.name

    #tf.io.gfile.makedirs(args.workdir) #不存在，则创建目录,tf阻碍logging屏幕输出

    if not os.path.isdir(args.workdir):
        os.makedirs(args.workdir)
    
    '''
       
    dist.init_process_group(backend="nccl")
    model = Runner(args)

    if args.test != 1:
        model.fit() #训练，验证和测试一体
    else: 
        model.test()#仅用于测试数据,需要配合args.save_path来使用
        
    dist.destroy_process_group()

    logging.info(('file name: %s') % (args.task_name + '_' + args.name)) #这个语句不会在标准输出上显示，但是会被写入文件
    print(('file name: %s') % (args.task_name + '_' + args.name))
    localtime = time.asctime(time.localtime(time.time()))
    logging.info(f'程序终止时间:{localtime}')
