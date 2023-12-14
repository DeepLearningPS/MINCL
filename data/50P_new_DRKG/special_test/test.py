import sys
import math
from tqdm import tqdm
import time
from collections import defaultdict
from ordered_set import OrderedSet
import copy
import difflib
import Levenshtein
import joblib
import os
import random
from collections import Counter
import math
import numpy

def lookup():
    file_list=['train.tsv',
               'type_count.txt',
               'all_node_to_types.txt',
               'new_node_to_types.txt',
               'new_category_text.txt',
               'new_entity2textlong.txt',
               'new_entity2name.txt',
               'relation2text.txt',
               'fb15_kg.tsv'
               
 
        ]
    
    for i in file_list[:1]:
        print(f'输出当前{i}的前30行：')
        count=0
        aa=set()
        aa1=set()
        with open(i,'r',encoding='utf-8')as f:
            
            for j in range(10):
                print(f.readline())





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



def count3():
    file_list=['train-ents-class.txt',
                'new_node_to_types.txt',
               'new_category_text.txt',
               'new_entity2textlong.txt',
               'new_entity2name.txt',
               'relation2text.txt'
               
 
        ]
    
    
    
    for file in file_list[:1]:
        old_set=set()
        new_set=set()
        count=0
        print(f'输出当前文件{file}：\n')
        if file == 'new_category_text.txt':
            with open(file,'r',encoding='utf-8')as f:
                for i in tqdm(f):
                    try:
                        tg=i.rstrip('\n').split('\t')
                        old_set.add(tg[0])
                        new_set.add(tuple(tg[1:]))
                    except Exception as e:
                        print(e)
                        print('error:',i)
                        if count>10:
                            exit()
                        else:
                            count+=1
                    
            print(f'{file}第一列:',len(old_set))
            print(f'{file}第二列:',len(new_set),'\n')

        else:   
            with open(file,'r',encoding='utf-8')as f:
                for i in tqdm(f):
                    try:
                        tg=i.rstrip('\n').split('\t')
                        old_set.add(tg[0])
                        new_set.add(tg[1])
                    except Exception as e:
                        print(e)
                        print('error:',i)
                        if count>10:
                            exit()
                        else:
                            count+=1
                        
            print(f'{file}第一列:',len(old_set))
            print(f'{file}第二列:',len(new_set),'\n')


def count4():
    file_list=['all_triples.tsv',
               'fb15_kg.tsv'
               
 
        ]
    
    
    count=0
    for file in file_list[:1]:
        h_set=set()
        r_set=set()
        t_set=set()
        print(f'输出当前文件{file}：\n')
        with open(file,'r',encoding='utf-8')as f:
            count=0
            for i in tqdm(f):
                try:
                    tg=i.rstrip('\n').split('\t')
                    try:
                        h_set.add(tg[0])
                        r_set.add(tg[1])
                        t_set.add(tg[2])
                    except:
                        
                        h_set.add(tg[2])
                        r_set.add(tg[3])
                        t_set.add(tg[4])
                    count+=1
                except Exception:
                    print('error:',i)
                    if count>10:
                        exit()
                    else:
                        count+=1
        print(f'{file}总数据:',count)          
        print(f'{file}第一列:',len(h_set))
        print(f'{file}第二列:',len(r_set))
        print(f'{file}第三列:',len(t_set),'\n')





def count6():
    '''
        看看训练集、验证集和测试集的class是否存在交叉
    '''
    node1={}
    node2={}
    node3={}
    node=set()
    rdf1={}
    rdf2={}
    rdf3={}
    rdf={}
    classes={}
    with open('train_class.txt')as f:
        for i in tqdm(f):
            tg=i.rstrip('\n').split('\t')
            node1[tg[0]]=tg[0]
            #node1[tg[2]]=tg[2]
            node.add(tg[0])
            #node.add(tg[2])
            rdf1[i]=i
            rdf[i]=i
            classes[tg[1]]=tg[1]

    with open('valid_class.txt')as f:
        for i in tqdm(f):
            tg=i.rstrip('\n').split('\t')
            node2[tg[0]]=tg[0]
            #node2[tg[2]]=tg[2]
            node.add(tg[0])
            #node.add(tg[2])
            rdf2[i]=i
            rdf[i]=i
            classes[tg[1]]=tg[1]

    with open('test_class.txt')as f:
        for i in tqdm(f):
            tg=i.rstrip('\n').split('\t')
            node3[tg[0]]=tg[0]
            #node3[tg[2]]=tg[2]
            node.add(tg[0])
            #node.add(tg[2])
            rdf3[i]=i
            rdf[i]=i
            classes[tg[1]]=tg[1]
    print('classes num:',len(classes))
    print('all node num:',len(node))
    print('train node num:',len(node1))
    print('train rdf_class num:',len(rdf1),'\n')
    print('dev node num:',len(node2))
    print('dev rdf_class num:',len(rdf2),'\n')
    print('test node num:',len(node3))
    print('test rdf_class num:',len(rdf3),'\n')
    
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

    '''
        结果：
    dev in train of node nums: 9801
    test in train of node nums: 10319

    dev in train of rdf nums: 0
    test in train of rdf nums: 0
    '''

    '''
    with open('new_class2id.txt','w')as f:
        for i,j in tqdm(enumerate(classes)):
            f.write(j+'\t'+str(i)+'\n')
    '''
    
    



        
def gen_type():
    '''
        category,短文本,我们使用word2vec嵌入，自己训练模型
    '''
                
    entity2type={}
    category_text={}
    with open('entity2type.txt')as f:
        for i in tqdm(f):
            tg=i.rstrip('\n').split('\t')[:3]#只取前2个type
            entity2type[tg[0]]=tg[1:]
    print('len(entity2type)：',len(entity2type),'\n')
    for i in entity2type:
        cs=[]
        text=[]
        for j in entity2type[i]:
            
            tg=j.split('/')[-1]
            cs.append(tg)
            text.append(' '.join(tg.split('_')))
            
            
        entity2type[i]=' '.join(cs)
        category_text[i]='\t'.join(text)

    with open('node_to_types.txt','w')as f:
        for i in tqdm(entity2type):
            f.write(i+'\t'+entity2type[i]+'\n')

    with open('category_text.txt','w')as f:
        for i in tqdm(category_text):
            f.write(i+'\t'+category_text[i]+'\n')

    bb=set()
    with open('node_to_types.txt')as f:
        for i in tqdm(f):
            tg=i.rstrip('\n').split('\t')
            for j in tg[1].split(' '):
                bb.add(j)
    print('type nums:',len(bb))
            
        

def gen_ture():

    kg={}
    with open('fb15_kg.tsv')as f:
        for i in tqdm(f):
            tg=i.rstrip('\n').split('\t')
            kg[tg[0]]=tg[0]
            kg[tg[2]]=tg[2]
            
    print('kG节点数量：',len(kg))

    with open('new_all_nodes.txt','w')as f:
        for i in kg:
            f.write(i+'\n')
        
    
    file_list=['node_to_types.txt',
               'category_text.txt',
               'entity2textlong.txt',
               'entity2text.txt'
               
 
        ]
    
    new_node_to_types=open('new_node_to_types.txt','w')
    new_category_text=open('new_category_text.txt','w')
    new_entity2textlong=open('new_entity2textlong.txt','w')
    new_entity2text=open('new_entity2text.txt','w')
    w_file=[new_node_to_types,new_category_text,new_entity2textlong,new_entity2text]
    #w_file=[new_node_to_types]
    for file,wf in zip(file_list[:],w_file[:]):
        with open(file)as f:
            count=0
            for i in f:
                tg = i.rstrip('\n').split('\t')
                if kg.get(tg[0]) != None:
                    wf.write(i)
                    count+=1

            print(f'文件{file}写入条数：',count)
                    
  
        

    new_node_to_types.close()
    new_category_text.close()
    new_entity2textlong.close()
    new_entity2text.close()
    

def look_loss():
    nodes={}
    with open('new_all_nodes.txt','r')as f:
        for i in f:
            tg=i.rstrip('\n')
            nodes[tg]=tg
    print('节点数量：',len(nodes))
    file_list=['new_node_to_types.txt',
               'new_category_text.txt',
               'new_entity2textlong.txt',
               'new_entity2text.txt'
               
 
        ]

    dict1={}
    dict2={}
    dict3={}
    dict4={}
    dict_list=[dict1,dict2,dict3,dict4]

    for file,dict_i in zip(file_list,dict_list):
        with open(file)as f:
            for i in f:
                tg=i.rstrip('\n').split('\t')
                dict_i[tg[0]]=tg[0]
                
    
    new_node_to_types=open('loss_new_node_to_types.txt','w')
    new_category_text=open('loss_new_category_text.txt','w')
    new_entity2textlong=open('loss_new_entity2textlong.txt','w')
    new_entity2text=open('loss_new_entity2text.txt','w')
    w_file=[new_node_to_types,new_category_text,new_entity2textlong,new_entity2text]
    #w_file=[new_node_to_types]
    
    for dict_i,wf in zip(dict_list[:],w_file[:]):
        count=0
        for i in nodes:
            if dict_i.get(i) == None:
                wf.write(i+'\n')
                count+=1
        print(f'文件{file}写入条数：',count)

            
    new_node_to_types.close()
    new_category_text.close()
    new_entity2textlong.close()
    new_entity2text.close()
    

def gen_type():
    aa=set()
    with open('new_node_to_types.txt')as f:
        for i in f:
            tg=i.rstrip('\n').split('\t')
            for j in tg[1].split(' '):
                aa.add(j)
    print('type num:',len(aa))

    with open('new_node_to_type.txt','w')as f:
        for i,j in enumerate(aa):
            f.write(j+str('i')+'\n')
        


def gen_all_types():

    entity2type={}
    #category_text={}
    with open('new_entity2type.txt')as f:
        for i in tqdm(f):
            tg=i.rstrip('\n').split('\t')[:]#只取前2个type
            entity2type[tg[0]]=tg[1:]
    print('len(entity2type)：',len(entity2type),'\n')
    for i in entity2type:
        cs=[]
        #text=[]
        for j in entity2type[i]:
            
            tg=j.split('/')[-1]
            cs.append(tg)
            #text.append(' '.join(tg.split('_')))
            
            
        entity2type[i]=' '.join(cs)
        #category_text[i]='\t'.join(text)

    with open('all_node_to_types.txt','w')as f:
        for i in tqdm(entity2type):
            f.write(i+'\t'+entity2type[i]+'\n')
        


def gen_count():
    #entity2type={}
    types=[]
    with open('all_node_to_types.txt')as f:
        for i in tqdm(f):
            tg=i.rstrip('\n').split('\t')[:]#只取前2个type
            for j in tg[1].split(' '):
                types.append(j)
    

    type_counts=open('type_count.txt','w',encoding='utf-8')
    count=0      
    type_count=Counter(types)#统计每一个type出现的频率,值得思考的是，这里统计的type频率的范围是多少呢？
    type_count=type_count.most_common()
    #所有节点，包括头尾实体
    
    for i,j in enumerate(type_count):#顺序混乱，问题就在这里，所以改成使用most_common()方法
        #xx.append(i)#type的id
        #yy.append(j[1])#type的频率
        type_counts.write(j[0]+'\t'+str(j[1])+'\n')
        count+=1
    type_counts.close()


def multi_relation():
    '''
        判断头尾节点对之间是否存在多个关系
    '''
    count=0
    h_t_dict=defaultdict(list)
    with open('all_triples.tsv')as f:
        for i in f:
            tg = i.rstrip('\n').split('\t')
            cs = str(tg[0])+str(tg[2])
            h_t_dict[cs].append(tg[1])

    for i in h_t_dict:
        if len(h_t_dict[i])>1:
            count+=1
    print('同一对节点之间存在多种关系的个数：',count)
            


def count_node():
    '''
        查看一下maintype是否是types的第一个元素的集合
    '''
    main_type = set()
    types = set()
    types_count = set()

    with open('maintype2id.txt')as f:
        for i in f:
            tg = i.split('\t')
            main_type.add(tg[0])

    with open('types2id.txt')as f:
        for i in f:
            tg = i.split('\t')[0].split(' ')[0]
            types.add(tg)


    with open('node2count.txt')as f:
        for i in f:
            tg = i.split('\t')[1].split(' ')[0]
            types_count.add(tg)
    print('len(main_type):',len(main_type))
    print('len(types2id):',len(types))
    print('len(types_count):',len(types_count))
    


def count(file_list):# 统计一下数据集的基本情况
    all_entity_set = OrderedSet()
    all_relation_set = OrderedSet()
    all_triple_set = OrderedSet()

    for file in file_list:
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
                
        print(file)       
        print('实体数量:',len(entity_set))
        print('关系数量:',len(relation_set))
        print('三元组数量:',len(triple_set))
        print('relation_set:', set(relation_set))
        print('\n')

    
    print('总的实体数量:',len(all_entity_set))
    print('总的关系数量:',len(all_relation_set))
    print('总的三元组数量:',len(all_triple_set)) 


if __name__ == '__main__':
    #count()
    
    #gen_type()
    #lookup()
    #count2()
    #count3()
    #count4()
    #count6()
    #gen_ture()
    #look_loss()
    #gen_type()
    #gen_all_types()
    #gen_count()
    #multi_relation()
    #count_node()
    #file_list = ['train.txt','valid.txt','test.txt']
    #count(file_list)
    
    file_list = []
    file_list = [file for file in os.listdir('../special_test') if '.txt' in file]
    count(file_list)