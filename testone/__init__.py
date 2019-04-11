#import requests
#r = requests.get(r"https://api.github.com/users/acombs/starred")
#r.json()

import os
import pandas as pd
import requests

PATH = r'./iris/'

#r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

#with open(PATH + 'iris.data','w') as f:
#    f.write(r.text)
    
#os.chdir(PATH) #改变当前工作目录到指定的路径

df = pd.read_csv(PATH + 'iris.data', names=['sepal length','sepal width','petal length','petal width','class'])

#以下是列表的基本转换语句
#print(df.head())
#print(df['sepal length'])
#print(df.iloc[:3,:2])
#print(df.loc[:3,[x for x in df.columns if 'width' in x]]) #因为带了标签 所以使用loc
#print(df['class'].unique())
#print(df[df['class'] == 'Iris-virginica'])  #保留了原始数据的行号
#print(df.count())
#print(df[df['class'] == 'Iris-virginica'].count())
#print(df[df['class'] == 'Iris-virginica'].reset_index(drop = True))  #重置了数据行号
#print(df[(df['class'] == 'Iris-virginica')&(df['petal width']>2.2)])
print(df.describe())
#print(df.describe(percentiles = [.20,.40,.80,.90,.95]))
print(df.corr()) 
#当协方差Cov(X,Y)>0时，称X与Y正相关    #当协方差Cov(X,Y)<0时，称X与Y负相关      #当协方差Cov(X,Y)=0时，称X与Y不相关
#Corr(X,Y)=1的时候，说明两个随机变量完全正相关，即满足Y=aX+b，a>0
#Corr(X,Y)=-1的时候，说明两个随机变量完全负相关，即满足Y=-aX+b，a>0

