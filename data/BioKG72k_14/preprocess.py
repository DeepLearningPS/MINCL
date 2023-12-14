import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import os
from collections import defaultdict
from ordered_set import OrderedSet
import copy
import os 
import shelve
import random

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", default='NoRepeat_PharmKG')
    args = args.parse_args()
    return args



def count4(files = None):

    all_trip_set = set()

    file_list=[
               #'NoRepeat_PharmKG.txt',
               #'drkg.tsv',
               #'NoRepeat_drkg.tsv',
               'train.txt',
               'valid.txt',
               'test.txt'
               
 
        ]
    


    if files != None:
        file_list = [files]
        
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
            
            
        

def gen_split(file_name):
    #划分数据集
    
    #df = pd.read_csv(file_name, sep='\t', header=None, nrows=1000000)  #PharmKG.csv自身有重复，所以去重复,如果没有表头，则加上header=None，或者加上表头,如果分割符不是','，则加上
    #该参数sep='\t'
    df = pd.read_csv(file_name, sep='\t', header=None)
    #df = df[: int(len(df) * 0.10)]

    

    #df = pd.read_csv(file_name, sep='\t', header=['h', 'r', 't']) 或者加上表头
    
    #df = df[['Entity1_ID', 'relation', 'Entity2_ID']] #如果没有表头，则注释掉
    #df = df[[0,1,2]] #如果没有表头，改成数字
    df = df.drop_duplicates() #去除重复的几列行数据
    train, test, _, _ = train_test_split(df, df, test_size=0.1, random_state=2020)
    train, valid, _, _ = train_test_split(train, train, test_size=0.11, random_state=2020)
    os.makedirs(args.dataset, exist_ok=True)
    train.to_csv(f'train.txt', sep='\t', index=False, header=None)
    valid.to_csv(f'valid.txt', sep='\t', index=False, header=None)
    test.to_csv(f'test.txt',   sep='\t', index=False, header=None)

    df.to_csv('sub_NoRepeat_BioKG.txt',  index = False, header = False, sep = '\t')
    df.to_csv('sub_NoRepeat_BioKG.tsv',  index = False, header = False, sep = '\t')
    

def cat_data():
    #将多个文件合并一起，且去重复
    all_data_list = []
    all_data_set  = OrderedSet()
    file_list = ['ddi_efficacy.tsv', 'ddi_minerals.tsv', 'dep_fda_exp.tsv', 'dpi_fda.tsv', 'phosphorylation.tsv']
    
    for file in file_list[:-1]:
        with open(file, 'r') as f:
            for i in f:
                all_data_list.append(i)
                all_data_set.add(i)
    
    with open(file_list[-1], 'r') as f:
        for i in f:
            tg = i.strip('\n').split('\t')
            cs = tg[:2] + [tg[2] + '_' + tg[3]]
            cs = '\t'.join(cs)
            all_data_list.append(cs + '\n')
            all_data_set.add(cs + '\n')
    
    print('all_data_num:', len(all_data_list))
    print('NoRepeat all_data_set_num:', len(all_data_set))
    #all_data_num: 1157739
    #NoRepeat all_data_set_num: 1157739
    
    with open('all_data.tsv', 'w') as f:
        for i in all_data_set:
            f.write(i)


if __name__ == '__main__':
    args = parse_args()
    args.dataset = '.'
    #count4('PharmKG.txt')
    #count4('sub_NoRepeat_drkg.txt')
    #count4()

    #cat_data() #合并数据
    
    gen_split('all_data.tsv')
    count4('sub_NoRepeat_BioKG.txt')
    count4()
    
    #gen_type('sub_NoRepeat_drkg.txt')
    #gen_relation()
    

    
    
    

    
    
