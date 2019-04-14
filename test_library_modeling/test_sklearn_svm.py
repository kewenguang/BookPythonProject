'''
Created on 2019年4月14日

@author: sgengine
'''
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split #导入将数据分成训练组和测试组的模块
import pandas as pd

PATH = r'./iris/'
df = pd.read_csv(PATH + 'iris.data', names=['sepal length','sepal width','petal length','petal width','class'])

clf = OneVsRestClassifier(SVC(kernel='linear'))

X = df.iloc[:,:4]
y = df.iloc[:,4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3) #train_test_split 将数据打乱并划分成四个子集 test_size=.3表示百分之三十的数据分配给X_test和y_test  
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
rf = pd.DataFrame(list(zip(y_pred, y_test)), columns=['predicted', 'actual']) #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
rf['correct'] = rf.apply(lambda r: 1 if r['predicted'] == r['actual'] else 0, axis = 1)
print(rf['correct'].sum()/rf['correct'].count())