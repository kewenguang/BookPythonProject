import matplotlib.pyplot as plt
import pandas as pd
from Cython.Shadow import inline
plt.style.use('ggplot')
#%matplotlib inline
import numpy as np

PATH = r'./iris/'
df = pd.read_csv(PATH + 'iris.data', names=['sepal length','sepal width','petal length','petal width','class'])

'''
#画一个直方图
fig, ax = plt.subplots(figsize=(6,4)) #6英寸,4英寸
ax.hist(df['petal width'], color='black')
ax.set_ylabel('Count',fontsize=12)
ax.set_xlabel('Width',fontsize=12)
plt.title('Iris Petal Width', fontsize=14, y=1.01)  #这个y控制着图表的标题应该显示在哪里
'''

'''
#画多个直方图
fig, ax = plt.subplots(2,2,figsize=(6,4))

ax[0][0].hist(df['petal width'], color='black')
ax[0][0].set_ylabel('Count', fontsize=12)
ax[0][0].set_xlabel('Width', fontsize=12)
ax[0][0].set_title('Iris Petal Width', fontsize=14, y=1.01)

ax[0][1].hist(df['petal length'], color='black')
ax[0][1].set_ylabel('Count', fontsize=12)
ax[0][1].set_xlabel('Width', fontsize=12)
ax[0][1].set_title('Iris Petal Length', fontsize=14, y=1.01)

ax[1][0].hist(df['sepal width'], color='black')
ax[1][0].set_ylabel('Count', fontsize=12)
ax[1][0].set_xlabel('Width', fontsize=12)
ax[1][0].set_title('Iris Sepal Width', fontsize=14, y=1.01)

ax[1][1].hist(df['sepal length'], color='black')
ax[1][1].set_ylabel('Count', fontsize=12)
ax[1][1].set_xlabel('Width', fontsize=12)
ax[1][1].set_title('Iris Sepal Length', fontsize=14, y=1.01)

plt.tight_layout()
'''

'''
#hist画直方图 scatter画散点图
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(df['petal width'],df['petal length'],color='green')
ax.set_xlabel('Petal Width')
ax.set_ylabel('Petal Length')
ax.set_title('Petal Scatter plot')
'''

'''
#plot把点连成线的图  通过这个点线图可以清楚地看到花瓣长度可能是分类的一个基准
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(df['petal length'], color='blue')
ax.set_xlabel('Specimen Number')
ax.set_ylabel('Petal Length')
ax.set_title('Petal Length Plot')
'''

fig, ax = plt.subplots(figsize=(6,6))
bar_width = .8
labels = [x for x in df.columns if 'length' in x or 'width' in x]
ver_y = [df[df['class'] == 'Iris-versicolor'][x].mean() for x in labels]
vir_y = [df[df['class'] == 'Iris-virginica'][x].mean() for x in labels]
set_y = [df[df['class'] == 'Iris-setosa'][x].mean() for x in labels]
x = np.arange(len(labels)) #x为[0 1 2 3]
ax.bar(x,vir_y,bar_width,bottom=set_y,color='darkgrey')
ax.bar(x,set_y,bar_width,bottom=ver_y,color='white')
ax.bar(x,ver_y,bar_width,color='black')
ax.set_xticks(x + (bar_width/2))
ax.set_xticklabels(labels,rotation=-70,fontsize=12)
ax.set_title('Mean Feature Measurement By Class', y=1.01)
ax.legend(['Virginica','Setosa','Versicolor'])

plt.show()
