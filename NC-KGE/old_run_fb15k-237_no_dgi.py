from helper import *
from data_loader import *
import sys
# sys.path.append('./')
from model.models_gat_no_dgi import *
from tqdm import tqdm
import os
import time
#多关系+对比
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
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
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
            for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, rel, obj))

                if split == 'train':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.data = dict(self.data)

        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = ddict(list)

        for (sub, rel), obj in self.sr2o.items():
            self.triples['train'].append({'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

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

        with open('./data/' + entity_file) as f:
            for line in f:
                entity_emb.append([float(val) for val in line.strip().split()])

        with open('./data/' + relation_file) as f:
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
        
        
        for arg in vars(args):
            self.logger.info((arg, getattr(args, arg)))#注意使用logger.info打印数据的时候，一定加一个括号，将数据当元组来看
        self.logger.info('\n\n')

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data()
        self.model = self.add_model(self.p.score_func)
        self.optimizer = self.add_optimizer(self.model.parameters())
        
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
            model = GAT_TransE(self.device, 400, 0.2, 512, non_linearity='prelu', params=self.p)
        elif score_func.lower() == 'distmult':
            model = GAT_DistMult(self.device, 400, 0.2, 512, non_linearity='prelu', params=self.p)
        elif score_func.lower() == 'conve':
            model = GAT_ConvE(self.device, 400, 0.2, 512, non_linearity='prelu', params=self.p)
        else:
            raise NotImplementedError

        model.to(self.device)
        num_param = sum([p.numel() for p in model.parameters()])
        print("Number of parameters: {0}".format(num_param)) #统计参数规模吗？可靠吗？怎么换算呢？等于多少M或者KB呢？
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
                    tri_loss = self.model.loss(pred, label)
                    loss = tri_loss + 0.001 * cl_loss
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
        change_lr = True #是否动态更改学习率
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
            tri_loss = self.model.loss(pred, label)
            loss = tri_loss + 0.001 * cl_loss

            if change_lr:
                #新的优化器，学习率自动变化
                self.optimizer.zero_grad() 
                torch.cuda.empty_cache()
                #loss.backward(retain_graph=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()
                
                if self.p.lr_scheduler in ['CyclicLR','OneCycleLR']:
                    self.scheduler.step()
                
            else:
                loss.backward()
                self.optimizer.step()

            losses.append(loss.item())
                
        if change_lr and self.p.lr_scheduler not in ['CyclicLR','OneCycleLR']:
            if self.p.lr_scheduler == 'ReduceLROnPlateau':  
                self.scheduler.step(loss)
            else:
                self.scheduler.step()
            
       
            
            '''
            if step % 100 == 0:
                self.logger.info('[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses), self.best_val_mrr, self.p.name))
            '''
        loss = np.mean(losses)
        #self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
        #print("[Epoch:{}]:  Training Loss:{:.4}\n".format(epoch, loss))
        return loss
    
    
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
            #train_loss = self.run_epoch(epoch, val_mrr) #把迭代放这里，比较好控制

            #change_lr = True #是否动态更改学习率
            self.model.train()
            losses = []
            train_iter = iter(self.data_iter['train'])
            iters = len(self.data_iter['train'])#获取批量数量，控制学习率的步长变化
            b_xent = nn.BCEWithLogitsLoss() #未使用
            lbl = torch.cat((torch.ones(1, self.p.num_ent), torch.zeros(1, self.p.num_ent)), 1).cuda() #未使用
            for step, batch in enumerate(train_iter):
                self.optimizer.zero_grad()
                sub, rel, obj, label = self.read_batch(batch, 'train')
                pred, cl_loss = self.model.forward(sub, rel)
                tri_loss = self.model.loss(pred, label)
                loss = tri_loss + 0.001 * cl_loss

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
                    
            if args.change_lr and self.p.lr_scheduler not in ['CyclicLR','OneCycleLR','CosineAnnealingWarmRestarts','ReduceLROnPlateau']:
                self.scheduler.step()
                
            train_loss = np.mean(losses)

            
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
        
        self.logger.info('Loading best model, Evaluating on Test data')
        self.load_model(save_path)
        #torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.name))#加载模型
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
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name',	default='rhcgat',					help='Set run name for saving/restoring models')
    parser.add_argument('--dataset',	dest='dataset',         default='FB15k-237',            help='Dataset to use, default: FB15k-237')
    parser.add_argument('--model',	dest='model',		default='rhcgat',		help='Model Name')
    parser.add_argument('--score_func',	dest='score_func',	default='conve',		help='Score Function for Link prediction')
    parser.add_argument('--opn',        dest='opn',             default='corr',                 help='Composition Operation to be used in CompGCN')

    parser.add_argument('--batch_size', dest='batch_size',      default=256,    type=int,       help='Batch size for GAT')
    parser.add_argument('--gamma',	type=float,             default=40.0,			help='Margin')
    parser.add_argument('--gpu',	type=str,               default='0',			help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--epoch',	dest='max_epochs', 	type=int,       default=2000,  	help='Number of epochs')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--lr',		type=float,             default=0.001,			help='Starting Learning Rate')
    parser.add_argument('--lbl_smooth', dest='lbl_smooth',	type=float,     default=0.1,	help='Label Smoothing')
    parser.add_argument('--num_workers',type=int,               default=10,                     help='Number of processes to construct batches')
    parser.add_argument('--seed',       dest='seed',            default=41504,  type=int,     	help='Seed for randomization')

    #parser.add_argument('--restore',   dest='restore',         action='store_true',            help='Restore from the previously saved model')
    parser.add_argument('--bias',       dest='bias',            default=0, type=int,            help='Whether to use bias in the model')
    #parser.add_argument('--hid_drop',   dest='hid_drop',        default=0.3, type=float, help='Dropout after GCN')
    parser.add_argument('--num_bases',	dest='num_bases', 	default=-1,   	type=int, 	help='Number of basis relation vectors to use')
    parser.add_argument('--init_dim',	dest='init_dim',	default=100,	type=int,	help='Initial dimension size for entities and relations')
    parser.add_argument('--gcn_dim',	dest='gcn_dim', 	default=200,   	type=int, 	help='Number of hidden units in GCN')
    parser.add_argument('--embed_dim',	dest='embed_dim', 	default=200,   type=int, 	help='Embedding dimension to give as input to score function')
    parser.add_argument('--gcn_layer',	dest='gcn_layer', 	default=2,   	type=int, 	help='Number of GCN Layers to use')
    parser.add_argument('--gcn_drop',	dest='dropout', 	default=0.1,  	type=float,	help='Dropout to use in GCN Layer')


    # arguments for GAT.不要使用bool类型了，作用和action='store_true',一样，如果指定值，则为真，否则使用默认值。使用0与1最好
    parser.add_argument('--pretrained_emb', dest='pretrained_emb', default=0, type=int,
                        help='')
    parser.add_argument('--num_of_layers',  dest='num_of_layers', default=2, type=int,
                        help='')
    parser.add_argument('--num_heads_per_layer', dest='num_heads_per_layer', nargs="*", default=[8, 1], type=int, help='')#两层网络，每一层的注意力头数量
    parser.add_argument('--add_skip_connection', dest='add_skip_connection', default=0, type=int, help='')
    parser.add_argument('--bias_gat',    dest='bias_gat', default=1, type=int, help='')
    parser.add_argument('--dropout_gat', dest='dropout_gat', default=0.5, type=float, help='')
    parser.add_argument('--num_features_per_layer', dest='num_features_per_layer', nargs="*",
                        default=[200, 25, 200], type=int, help='') #25是每一个注意力头的维度，即200/8 = 25
    
    #改一下注意力头和网络层的执行，要更灵活，而不是只能执行2层网络。

    # ConvE specific hyperparameters
    parser.add_argument('--hid_drop',   dest='hid_drop',    default=0.3,        type=float,     help='ConvE1: Hidden dropout')
    parser.add_argument('--hid_drop2',  dest='hid_drop2',   default=0.4,  	type=float,	help='ConvE2: Hidden dropout')
    parser.add_argument('--feat_drop', 	dest='feat_drop',   default=0.3,  	type=float,	help='ConvE: Feature Dropout')
    parser.add_argument('--k_w',	dest='k_w', 	    default=10,   	type=int, 	help='ConvE: k_w')
    parser.add_argument('--k_h',	dest='k_h', 	    default=20,   	type=int, 	help='ConvE: k_h')
    parser.add_argument('--num_filt',  	dest='num_filt',    default=200,   	type=int, 	help='ConvE: Number of filters in convolution')
    parser.add_argument('--ker_sz',    	dest='ker_sz', 	    default=8,   	type=int, 	help='ConvE: Kernel size to use')

    #parser.add_argument('--logdir',          dest='log_dir',         default='./log/',               help='Log directory')
    #parser.add_argument('--config',          dest='config_dir',      default='./config/',            help='Config directory')


    #一些新参数

    parser.add_argument('--local_rank', default=0, type=int,
                            help='node rank for distributed training')
    
    parser.add_argument('--use_pretrain', type=int, default=0,
                        help='whether use pretrain')

    parser.add_argument('--use_test', type=int, default=0,
                        help='whether be used in test')

    parser.add_argument('--test', type=int, default=0,
                        help='whether be used in testdata') #配合save_path使用，仅用于测试数据集

    parser.add_argument('--test_name', type=str, default='',
                        help='test name')

    parser.add_argument('--model_dir', type=str, default='./fb15k237_model_save',
                        help='The address for storing the models and optimization results.')

    parser.add_argument('--task_name', type=str, default='LP',
                        help='The name of the stored models and optimization results.')

    parser.add_argument('--logdir', dest='log_dir', default='./fb15k237_log/', help='Log directory')

    parser.add_argument('--config', dest='config_dir', default='./config/', help='Config directory')
            
    parser.add_argument('--loss_function', type=str, default='bceloss',
                        choices=['contrastive_loss', 'mask_softmax', 'mix_loss','bceloss'],
                        help='choose loss function')#改一下,多标签损失

    parser.add_argument('--stop_num', type=int, default=50,
                        help='stop num of epoch')

    parser.add_argument('--save_path', type=str, default='',
                        help='load model from save path')

    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam', 'sgd', 'adagrad'],#学习率的优化，注意所给的学习率优化器必须在choice里面
                        help='optimizer to use.')
    
    parser.add_argument('--change_lr', default=1, type=int,
                        help='change lr') #优先使用动态调整学习率的方法，找到最优之后再使用静态的学习率

    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR',
                        choices=['CyclicLR', 'OneCycleLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau', 'CosineAnnealingLR'],#学习率的优化，注意所给的学习率优化器必须在choice里面
                        help='lr scheduler to use.')
    
    parser.add_argument('--clip', type=float, default=0.25,
                    help='Gradient Norm Clipping') #梯度修剪，防止梯度爆炸吗？是的
    
    parser.add_argument('--conv_name', type=str, default='rhcgat',#根据不同的模型名，执行不同的模型,这个看看如何切换
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
    args = parser.parse_args()


    if args.use_pretrain:
        if args.use_test:
            name = args.test_name + '_' + args.dataset + '_'+ 'CL'+ args.conv_name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H_%M_%S')
        else:
            name = args.dataset + '_'+ 'CL'+ args.conv_name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H_%M_%S')
    else:
        if args.use_test:
            name = args.test_name + '_' + 'notext_' + args.dataset + '_'+ 'CL'+ args.conv_name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H_%M_%S')
        else:
            name = 'notext_' + args.dataset + '_'+ 'CL'+ args.conv_name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H_%M_%S')

    #args.name = name#新增参数
    

    args.name = name#新增参数

    set_gpu(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = Runner(args)

    if args.test != 1:
        model.fit() #训练，验证和测试一体
    else: 
        model.test()#仅用于测试数据,需要配合args.save_path来使用
