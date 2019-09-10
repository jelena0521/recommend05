#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 15:10:45 2019

@author: liujun
"""


import pandas as pd
import pickle
import numpy as np
import random
import time
from math import exp



class LFM:
    def __init__(self):
        self.lr=0.02 
        self.lam=0.01
        self.class_count=5
        self.iter_count=5
        self.init_model()
        
        
        
    def init_model(self):
        file_path='ratings.csv' #userid,itemid,rate
        pos_neg_path='pni.dict'
        self.user_movie_rate=pd.read_csv(file_path)
        self.user_ids=set(self.user_movie_rate['userid'].values)
        self.item_ids=set(self.user_movie_rate['movieid'].values)
        self.items_dict=pickle.load(open(pos_neg_path,'rb'))
        array_p=np.random.randn(len(self.user_ids), self.class_count) #初始化矩阵 行*列 用户ID数为行数 隐分类类别为列数
        array_q=np.random.randn(len(self.item_ids), self.class_count) #初始化矩阵 行*列 物品ID数为行数 隐分类类别为列数
        self.p=pd.DataFrame(array_p,columns=range(0, self.class_count),index=list(self.user_ids))
        self.q=pd.DataFrame(array_q,columns=range(0, self.class_count),index=list(self.item_ids))
     
    def predict(self,user_id,item_id):
        p=np.mat(self.p.ix[user_id].values)
        q=np.mat(self.q.ix[item_id].values).T
        r=(p*q).sum() #矩阵相乘
        logit=1/(1+exp(-r)) #sigmoid函数 (0,1) 用户u对物品i的感兴趣值
        return logit
    
    def loss(self,user_id,item_id,y,step):
        e=y-self.predict(user_id,item_id) #e为损失函数 
        return e
        


    def optimize(self,user_id,item_id,e):
        gradient_p=-e*self.q.ix[item_id].values #梯度求导
        l2_p=self.lam*self.p.ix[user_id].values #l2正则 防止过拟合
        delta_p=self.lr*(gradient_p+l2_p) #乘上学习率
        
        gradient_q=-e*self.p.ix[user_id].values
        l2_q=self.lam*self.q.ix[item_id].values
        delta_q=self.lr*(gradient_q+l2_q)
        
        self.p.loc[user_id]=self.p.loc[user_id]-delta_p #跟新行的值
        self.q.loc[item_id]=self.q.loc[item_id]-delta_q #更新列的值
        
    def train(self):
            for step in range(0,self.iter_count): #迭代次数
                time.sleep(30)
                for user_id,item_dict in self.items_dict.items(): #{userid:{itemid:1/0...}...}
                    item_ids=list(item_dict.keys()) #所有的itemid
                    random.shuffle(item_ids)
                    for item_id in item_ids:
                        e=self.loss(user_id,item_id,item_dict[item_id],step)
                        self.optimize(user_id,item_id,e)
                self.lr=self.lr*0.9 #衰减学习率
            self.save()  #save的是两个矩阵 user*类别 和 item*类别
            
            
    def save(self):
        f=open('lfm.model','wb')
        pickle.dump((self.p,self.q),f)
        f.close()
        
    def load(self):
        f=open('lfm.model','rb')
        self.p,self.q=pickle.load(f)
        f.close()
        
    def predict_train(self,user_id,top_n=10):
        self.load()
        user_item_ids=set(self.item_rating_csv[self.item_rating_csv['userid']==user_id]['movieid']) #该用户评价的所有itemid的集合
        other_item_ids=self.item_ids^user_item_ids #该用户未评价的所有itemID的集合
        interest_list=[self.predict(user_id,item_id) for item_id in other_item_ids] #得到未评价的物品的兴趣值
        candidates=sorted(zip(list(other_item_ids),interest_list),key=lambda x:x[1],reverse=True) #对兴趣值排序
        return candidates[:top_n]
        
    def evaluate(self):
        self.load()
        users=random.sample(self.user_ids,10) #随机取10个
        user_dict={}
        for user in users:
            user_item_ids=set(self.item_rating_csv[self.item_rating_csv['userid']==user]['movieid']) #该用户评价过的物品的集合
            eva_sum=0
            for item_id in user_item_ids:
                p=np.mat(self.p.ix[user].values)
                q=np.mat(self.q.ix[item_id].values)
                r=(p*q).sum() #预测值
                z=self.item_rating_csv[(self.item_rating_csv['userid']==user)
                & (self.item_rating_csv['movieid']==item_id)]['rating'].values[0] #实际值
                eva_sum=eva_sum+abs(r-z)
                user_dict[user]=eva_sum/len(user_item_ids) #总差值/总物品数
                print('userid:{},mse:{}'.format(user,user_dict[user]))
        return sum(user_dict.values())/len(user_dict.keys()) #10个的总差值/10个的总物品数
    
if __name__=='__main__':
    lfm=LFM()
    lfm.train()
    print(lfm.evaluate())

                
              
    
        
        
                
                