import pandas as pd
import seaborn as sns

PATH = r'./iris/'
df = pd.read_csv(PATH + 'iris.data', names=['sepal length','sepal width','petal length','petal width','class'])
sns.pairplot(df, hue="class")