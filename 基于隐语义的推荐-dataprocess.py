#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:17:21 2019

@author: liujun
"""

import os
import pandas as pd
import pickle

class dataprocessing:
    def __init__(self):
        pass
    
    def process(self):
        print('预处理转换用户数据')
        self.process_user_data()
        print('预处理评分数据')
        self.process_rating_data()
        print('预处理电影数据')
        self.process_movies_data
    
    def process_user_data(self,file='users.dat'):
        if not os.path.exists('users.csv'):
            fp=pd.read_table(file,sep="::",names=['userid','gender','age','occupation','zipcode'])
            fp.to_csv('users.csv',index=False)
            
    def process_rating_data(self,file='ratings.dat'):
        if not os.path.exists('ratings.csv'):
            fp=pd.read_table(file,sep='::',names=['userid','movieid','rating','timestamp'])
            fp.to_csv('ratings.csv',index=False)
    
    def process_movies_data(self,file='movies.dat'):
        if not os.path.exists('movies.csv'):
            fp=pd.read_table(file,sep="::",names=['movieid','title','genres'])
            fp.to_csv('movies.csv',index=False)
            
    def get_pos_neg_item(self,file='ratings.csv'): #生成的文件表示为，该用户有行为为1，没有则为0
        if not os.path.exists('pni.dict'):
            self.items_rating_path='pni.dict'
            self.item_rating_csv=pd.read_csv(file)
            self.userids=set(self.item_rating_csv['userid'].values) #所有的userid集合
            self.movieids=set(self.item_rating_csv['movieid'].values) #所有的itemid集合
            self.user_dict={userid:self.get_one(userid) for userid in list(self.userids)} #{userid:{itemid1:1/0...}}所有的物品
            fw=open(self.items_rating_path,'wb')
            pickle.dump(self.user_dict,fw)
            fw.close()
    
    def get_one(self,user_id): #该用户有行为为1 没有行为为0
       pos_item_ids=set(self.item_rating_csv[self.item_rating_csv['userid']==user_id]['movieid']) #该用户评价的itemID的集合
       neg_item_ids=self.movieids^pos_item_ids
       item_dict={}
       for item in pos_item_ids:item_dict[item]=1
       for item in neg_item_ids:item_dict[item]=0
       return item_dict

if __name__=='__main__':
    dp=dataprocessing()
    dp.process()
    dp.get_pos_neg_item()


   





