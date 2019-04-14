'''
Created on 2019年4月14日

@author: sgengine
'''
import os
import pandas as pd
import requests
import matplotlib.pyplot as plt

PATH = r'./iris/'
df = pd.read_csv(PATH + 'iris.data', names=['sepal length','sepal width','petal length','petal width','class'])

#先画散点图观察一下数据大概走势
fig, ax = plt.subplots(figsize=(7,7))
ax.scatter(df['sepal width'][:50], df['sepal length'][:50])
ax.set_ylabel('Sepal Length')
ax.set_xlabel('Sepal Width')
ax.set_title('Setose Sepal Width vs. Sepal Length', fontsize=14,y=1.02)
#plt.show()

#利用观察的大概样子建模
import statsmodels.api as sm
y = df['sepal length'][:50]
x = df['sepal width'][:50]
X = sm.add_constant(x)
results = sm.OLS(y, X).fit()
print(results.summary())

#把建立的模型画出来
ax.plot(x, results.fittedvalues, label='regression line')
ax.scatter(x, y, label='data point', color='r')  #画点也画标签
ax.set_ylabel('Sepal Length')
ax.set_xlabel('Sepal Width')
ax.set_title('Setosa Sepal Width vs. Sepal Length', fontsize=14,y=1.02)
ax.legend(loc=2) #加了这句话，红色的点代表啥，蓝色的线代表啥，会在角落里加上一些标签以表明含义 1代表右上角 2代表左上角 3代表左下角 4代表右下角

plt.show()