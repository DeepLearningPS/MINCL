import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import os
from collections import defaultdict
from ordered_set import OrderedSet
import copy
import os 
import shelve
import torch
import numpy as np
import random
import os
import shutil
from tqdm import tqdm

np.random.seed(2023)
torch.manual_seed(2023)
random.seed(2023)
torch.cuda.manual_seed_all(2023)



def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", default='NoRepeat_PharmKG')
    args = args.parse_args()
    return args




def count2():
    '''
        看看训练集、验证集和测试集是否存在交叉
    '''
    node1={}
    node2={}
    node3={}
    node=set()
    rdf1={}
    rdf2={}
    rdf3={}
    rdf={}
    rdf_list=[]
    with open('train.txt')as f:
        for i in tqdm(f):
            tg=i.rstrip('\n').split('\t')
            node1[tg[0]]=tg[0]
            node1[tg[2]]=tg[2]
            node.add(tg[0])
            node.add(tg[2])
            rdf1[i]=i
            rdf[i]=i
            rdf_list.append(i)

    with open('valid.txt')as f:
        for i in tqdm(f):
            tg=i.rstrip('\n').split('\t')
            node2[tg[0]]=tg[0]
            node2[tg[2]]=tg[2]
            node.add(tg[0])
            node.add(tg[2])
            rdf2[i]=i
            rdf[i]=i
            rdf_list.append(i)

    with open('test.txt')as f:
        for i in tqdm(f):
            tg=i.rstrip('\n').split('\t')
            node3[tg[0]]=tg[0]
            node3[tg[2]]=tg[2]
            node.add(tg[0])
            node.add(tg[2])
            rdf3[i]=i
            rdf[i]=i
            rdf_list.append(i)
    print('总的RDF个数：',len(rdf))
    print('总的不去重复RDF个数：',len(rdf_list))
    
    print('all node num:',len(node))
    print('train node num:',len(node1))
    print('train rdf num:',len(rdf1),'\n')
    print('dev node num:',len(node2))
    print('dev rdf num:',len(rdf2),'\n')
    print('test node num:',len(node3))
    print('test rdf num:',len(rdf3),'\n')
    '''
    train_loss=open('train_loss.txt','w')
    count = 0
    for ii in node:
        if ii not in node1:
            train_loss.write(ii+'\n')
            count += 1
    print('train_loss:',count)
    '''
            
    
    count=0
    for i in node2:
        if node1.get(i) != None:
            count+=1
    print('dev in train of node nums:',count)

    count=0
    for i in node3:
        if node1.get(i) != None:
            count+=1
    print('test in train of node nums:',count,'\n')

    count=0
    for i in rdf2:
        if rdf1.get(i) != None:
            count+=1
    print('dev in train of rdf nums:',count)

    count=0
    for i in rdf3:
        if rdf1.get(i) != None:
            count+=1
    print('test in train of rdf nums:',count)


    count=0
    for i in node3:
        if node2.get(i) != None:
            count+=1
    print('test in valid of node nums:',count,'\n')
    
    count=0
    for i in rdf3:
        if rdf2.get(i) != None:
            count+=1
    print('test in valid of rdf nums:',count)

    '''
        结果：
    dev in train of node nums: 9801
    test in train of node nums: 10319

    dev in train of rdf nums: 0
    test in train of rdf nums: 0
    '''
    '''
    with open('fb15_kg.tsv','w')as f:
        for i in tqdm(rdf):
            f.write(i)
    '''




def count4(files = None):

    all_trip_set = set()

    file_list=[
               #'NoRepeat_PharmKG.txt',
               #'drkg.tsv',
               'sub_NoRepeat_drkg.tsv',
               'train.txt',
               'valid.txt',
               'test.txt'
               
 
        ]
    


    if files != None:
        file_list = files
        
    first = ''
    for file in file_list[:]:
    
        
        h_set=OrderedSet()  # entity_826,Chemical,entity_5788,Gene,relation_26
        r_set=OrderedSet()
        t_set=OrderedSet()
        node_set  = OrderedSet()
        count_set = OrderedSet()
        print(f'输出当前文件{file}：\n')
        with open(file, 'r', encoding='utf-8')as f:
            #first = f.readline()
            count=0
                
            for i in f:
                count_set.add(i.rstrip('\n'))
                tg=i.rstrip('\n').split('\t')
                        
                h_set.add(tg[0])
                r_set.add(tg[1])
                t_set.add(tg[2])
                node_set.add(tg[0])
                node_set.add(tg[2])
                all_trip_set.add(i.rstrip('\n'))
                #all_trip_set.add(tg[2])
                
                count+=1

        print(f'{file} 总数据:',count)
        print(f'{file} 去重复后的总数据:',len(count_set))   #PharmKG.csv自身有重复
        print(f'{file} node_num:',len(node_set))
        print(f'{file} h_set:',   len(h_set))
        print(f'{file} r_set:',   len(r_set))
        print(f'{file} t_set:',   len(t_set), '\n')

        #all_trip_set = all_trip_set | count_set #这是没问题的


        if count != len(count_set):
            print('count != len(count_set)')
            with open('NoRepeat_' + file, 'w')as f:
                f.write(first)
                for i in count_set:
                    f.write(i + '\n')

    print('len(all_trip_set):', len(all_trip_set))


def gen_type(files = None):

    file_list=['NoRepeat_drkg.tsv',
                      
        ]
    


    if files != None:
        file_list = [files]
        
    first = ''
    for file in file_list[:]:
        
        h_set=set()  # entity_826,Chemical,entity_5788,Gene,relation_26
        r_set=set()
        t_set=set()
        node_type_set = set()
        count_set = set()
        node_type_dict = {}
        print(f'输出当前文件{file}：\n')
        with open(file,'r',encoding='utf-8')as f:
            #first = f.readline()
            #print(first)
            count=0
            for i in f:
                count_set.add(i.rstrip('\n'))
                
                tg=i.rstrip('\n').split('\t')
            
                h_set.add(tg[0])
                r_set.add(tg[1])
                t_set.add(tg[2])
                node_type_set.add(tg[0].split('::')[0])
                node_type_set.add(tg[2].split('::')[0])
                node_type_dict[tg[0]] = tg[0].split('::')[0]
                node_type_dict[tg[2]] = tg[2].split('::')[0]

                count+=1
        print(f'{file} 总数据:',count)
        print(f'{file} 去重复后的总数据:',len(count_set))   #PharmKG.csv自身有重复
        print(f'{file} h_set:',len(h_set))
        print(f'{file} r_set:',len(r_set))
        print(f'{file} t_set:',len(t_set))
        print(f'{file} node_type_set:',len(node_type_set),'\n')
        print('len(node_type_dict):', len(node_type_dict))

    type2id = {}
    id2type = {}
    with open(f'type2id.txt', 'w')as f:
        for ids, tp in enumerate(node_type_set):
            type2id[tp] = ids
            id2type[ids]= tp
            f.write(tp + '\t' + str(ids) + '\n')
            

    with open(f'node2type.txt', 'w')as f:
        for n_key in node_type_dict:
            f.write(n_key + '\t' + node_type_dict[n_key] + '\n')
            
    
    with open(f'node2typeid.txt', 'w')as f:
        for n_key in node_type_dict:
            f.write(n_key + '\t' + str(type2id[node_type_dict[n_key]]) + '\n')





def gen_relation():
    #生成关系ID以及关系的描述

    relation2id  = {}
    relation2des = {}

    re_set1 = OrderedSet() #关系名
    re_set2 = OrderedSet()
    re_set3 = OrderedSet()
    re_set4 = OrderedSet()
    re_dict5 = {} #描述

    re_list1 = []
    re_list2 = []
    re_list3 = []
    re_list4 = []
    re_list5 = []
    
    
    file =  'relation_glossary.tsv'
    print(f'输出当前文件{file}：\n')

    with open('relation_glossary.tsv') as f:
        head = f.readline().split('\t')

        for i in f:
            tg = i.rstrip('\n').split('\t')
            #print(len(tg)) #6个字段
            #print(tg)
            #exit()
            re_set1.add(tg[0])
            re_set2.add(tg[1])
            re_set3.add(tg[2])
            re_set4.add(tg[3])
            re_dict5[tg[0]] = tg[4]

            re_list1.append(tg[0])
            re_list2.append(tg[1])
            re_list3.append(tg[2])
            re_list4.append(tg[3])
            re_list5.append(tg[4])
            

    for i, j, k in zip(head, [re_set1, re_set2, re_set3, re_set4, re_dict5], [re_list1, re_list2, re_list3, re_list4, re_list5]):
        print(f'{i}, NoRepeat, Repeat: {len(j)}, {len(k)}')

    with open('relation2id.txt', 'w')as f:
        for ids, j in enumerate(re_set1):
            f.write(j + '\t' + str(ids) + '\n')

    with open('relation2description.txt', 'w')as f:
        for key in re_dict5:
            f.write(key + '\t' + re_dict5[key] + '\n')
            
            



def gen_subdata(file):  #NoRepeat_drkg.tsv, 保存成sub_NoRepeat_drkg.tsv
    #生成子集
    data_dict = defaultdict(list) #存放每一种关系对应三元组

    with open(file) as f:  
        for i in f:
            tg = i.strip('\n').split('\t')
            if len(tg) != 3:
                raise Exception(f'{len(tg)} != 3')
            data_dict[tg[1]].append(i)

    
    sub_data_list = []

    for r in data_dict:
        dt = copy.deepcopy(data_dict[r])
        random.shuffle(dt)
        end_index = round(len(dt) * 0.1) #取打乱之后的前10%
        sub_data_list.extend(dt[:end_index])

    print('len(sub_data_list) = {}'.format(len(sub_data_list)))
          
    with open(f'sub_{file}', 'w')as f:
        for i in sub_data_list:
            f.write(i)
    



def gen_split(file_name):
    #划分数据集
    
    df = pd.read_csv(file_name, sep='\t', header=None)  #PharmKG.csv自身有重复，所以去重复,如果没有表头，则加上header=None，或者加上表头,如果分割符不是','，则加上
    #该参数sep='\t'
    #df = pd.read_csv(file_name, sep='\t', header=['h', 'r', 't']) 或者加上表头
    
    #df = df[['Entity1_ID', 'relation', 'Entity2_ID']] #如果没有表头，则注释掉
    #df = df[[0,1,2]] #如果没有表头，改成数字
    df = df.drop_duplicates() #去除重复的几列行数据
    #print('df:', df[3])
    train, test, _, _ = train_test_split(df, df, test_size=0.1, random_state=2023)
    train, valid, _, _ = train_test_split(train, train, test_size=0.11, random_state=2023)
    os.makedirs(args.dataset, exist_ok=True)
    train.to_csv(f'train.txt', sep='\t', index=False, header=None)
    valid.to_csv(f'valid.txt', sep='\t', index=False, header=None)
    test.to_csv(f'test.txt',   sep='\t', index=False, header=None)





def gen_subdata_train():
    #将训练集分成50%，75%，100%。 由于当前是100%，所以这个不用生成了。

    os.makedirs(f'../50P_new_DRKG/', exist_ok=True)
    os.makedirs(f'../75P_new_DRKG/', exist_ok=True)
    os.makedirs(f'../100P_new_DRKG/', exist_ok=True)


    # 源文件路径
    source_file1 = 'valid.txt'
    source_file2 = 'test.txt'
    # 目标文件夹路径
    destination_folder1 = '../50P_new_DRKG/'
    destination_folder2 = '../75P_new_DRKG/'
    destination_folder3 = '../100P_new_DRKG/'

    # 将文件复制到目标文件夹
    shutil.copy(source_file1, destination_folder1)
    shutil.copy(source_file2, destination_folder1)
    shutil.copy(source_file1, destination_folder2)
    shutil.copy(source_file2, destination_folder2)
    shutil.copy(source_file1, destination_folder3)
    shutil.copy(source_file2, destination_folder3)

    
    #划分训练集
    data     = []
    data_50  = []
    data_75  = []
    data_100 = []
    
    with open('train.txt') as f:
        for i in f:
            data.append(i)
    

    dt = copy.deepcopy(data)
    random.shuffle(dt)
    end_index1 = round(len(dt) * 0.5)  #取打乱之后的前50%
    end_index2 = round(len(dt) * 0.75) #取打乱之后的前75%
    end_index3 = round(len(dt) * 1)    #取打乱之后的前100%
    data_50.extend(dt[:end_index1])
    data_75.extend(dt[:end_index2])
    data_100.extend(dt[:end_index3])

    with open(destination_folder1 + 'train.txt', 'w') as f:
        for i in data_50:
            f.write(i)
    

    with open(destination_folder2 + 'train.txt', 'w') as f:
        for i in data_75:
            f.write(i)


    with open(destination_folder3 + 'train.txt', 'w') as f:
        for i in data_100:
            f.write(i)

            





def gen_alldata(file_list): #生成所有实体名和关系名到id
    relation_set = OrderedSet()
    entity_set = OrderedSet()
    all_data = []
    for file in file_list:
        with open(file) as f:
            for i in f:
                all_data.append(i)
                tg = i.rstrip('\n').split('\t')
                if len(tg) != 3:
                    raise Exception('len(tg) != 3')
                relation_set.add(tg[1])
                entity_set.add(tg[0])
                entity_set.add(tg[2])

    print('len(relation_set):',len(relation_set))
    print('len(entity_set):',len(entity_set))
    
    print('len(all_data):',len(all_data))

    with open('all_data.txt', 'w') as f:
        for i in all_data:
            f.write(i)

    


if __name__ == '__main__':
    args = parse_args()
    args.dataset = '.'

    
    #count4(['NoRepeat_drkg.tsv'])  #统计原始的数据


    #gen_subdata('NoRepeat_drkg.tsv') #生成10%的子集
    
    #gen_split('sub_NoRepeat_drkg.tsv') #划分训练、验证、测试
    #gen_type('all_data.txt')  #如果有节点类型，则生成
    #gen_relation() #这个用不着
    
    #count2()  #统计训练验证测试是否交叉

    file = [ 'sub_NoRepeat_drkg.tsv',
            'train.txt',
            'valid.txt',
            'test.txt'
            ]
    #count4(file[:])


    #gen_subdata_train()


    file_list = ['train.txt','valid.txt','test.txt']
    gen_alldata(file_list) #合并所有的数据

    gen_type('all_data.txt') 
    

    
    
    

    
    
