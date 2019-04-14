'''
Created on 2019年4月14日

@author: sgengine
'''
from sklearn.ensemble import RandomForestClassifier  #这个导入的是随机森林分类器
from sklearn.model_selection import train_test_split #导入将数据分成训练组和测试组的模块
import pandas as pd

PATH = r'./iris/'
df = pd.read_csv(PATH + 'iris.data', names=['sepal length','sepal width','petal length','petal width','class'])

clf = RandomForestClassifier(max_depth=5,n_estimators=10)  #一个使用十个决策树的森林 每棵树最多允许5层判定深度

X = df.iloc[:,:4]
y = df.iloc[:,4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3) #train_test_split 将数据打乱并划分成四个子集 test_size=.3表示百分之三十的数据分配给X_test和y_test  
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
rf = pd.DataFrame(list(zip(y_pred, y_test)), columns=['predicted', 'actual']) #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
rf['correct'] = rf.apply(lambda r: 1 if r['predicted'] == r['actual'] else 0, axis = 1)
print(rf['correct'].sum()/rf['correct'].count())
#print(rf)

#查看哪些特征提供了最佳辨别力或者预测能力
'''
import numpy as np
f_importances = clf.feature_importances_f_names = df.columns[:4]
f_std = np.std([tree.feature_importances_ for tree in clf.estimators], axis=0)
f_name = ['花瓣长度','花瓣宽度','花萼长度','花萼宽度']
zz = zip(f_importances, f_name, f_std)
zzs = sorted(zz, key=lambda x:x[0], reverse=True)
imps = [x[0] for x in zzs]
labels = [x[1] for x in zzs]
errs = [x[2] for x in zzs]
import matplotlib.pyplot as plt
plt.bar(range(len(f_importances)), imps, color="r", yerr=errs, align="center")
plt.xticks(range(len(f_importances)),labels)
plt.show()'''