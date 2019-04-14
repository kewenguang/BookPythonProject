import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

PATH = r'./iris/'
df = pd.read_csv(PATH + 'iris.data', names=['sepal length','sepal width','petal length','petal width','class'])

#这个是可以打出很多变量关系图的
#sns.pairplot(df, hue="class")

#这个是可以画violin的图，通过看那个地方比较宽大可以看出数据在哪个范围比较集中
'''
fig, ax = plt.subplots(2, 2, figsize=(7,7))

sns.set(style='white', palette='muted')
sns.violinplot(x=df['class'], y=df['sepal length'], ax=ax[0,0])
sns.violinplot(x=df['class'], y=df['sepal width'], ax=ax[0,1])
sns.violinplot(x=df['class'], y=df['petal length'], ax=ax[1,0])
sns.violinplot(x=df['class'], y=df['petal width'], ax=ax[1,1])
fig.suptitle('Violin Plots', fontsize=16, y=1.03)
for i in ax.flat:
    plt.setp(i.get_xticklabels(), rotation=-90)
fig.tight_layout()
plt.show()
'''

#原来的数据按照Iris-setosa Iris-virginica Iris-versicolor来分类的，现在按照简称分类SET  VIR VER
#实现了一个简单的名称映射
#df['class'] = df['class'].map({'Iris-setosa':'SET', 'Iris-virginica':'VIR', 'Iris-versicolor':'VER'})
#print(df)

#相当于一个SQL查询，对表加条件生成新的列
#df['wide petal'] = df['petal width'].apply(lambda v:1 if v >= 1.3 else 0)
#print(df)

#在整个数据框的数据分析上加列
#df['petal area'] = df.apply(lambda r:r['petal length'] * r['petal width'], axis=1)
#print(df)

#对每一个数据单元进行处理，得到最终的一个数据表
'''
import numpy as np
df = df.applymap(lambda v: np.log(v) if isinstance(v, float) else v)
print(df)
'''

#按照class对特征进行分类，然后对每个特征求平均值
'''
df = df.groupby('class').mean()
print(df)
'''

#按照class对特征进行分类，并对特征求平均值，求标准差，求最小值等等
#df = df.groupby('class').describe()
#print(df)

#按照petal width进行分类，并标出宽度为这个值的类有哪些
#df = df.groupby('petal width')['class'].unique().to_frame()
#print(df)

#import numpy as np
#df = df.groupby('class')['petal width'].agg({'delta':lambda x:x.max() - x.min(),'max':np.max, 'min':np.min})
#print(df)

#以上只是一些简单的groupby功能，更多的要看下面的URL
#http://pandas.pydata.org/pandas-docs/stable/
