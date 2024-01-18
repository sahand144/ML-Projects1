import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from keras.utils import plot_model
import seaborn as sns
import matplotlib.pyplot as plt

from LinearReg import X_test, X_train 

#importing dataset
datadir = r"D:\datasets\DecisionTree\titanic.csv"
df = pd.read_csv(datadir)
df.head()
df.dtypes

#find categorical columns and store them in a list
categorical_columns = []
for col in df.columns:
    if df[col].dtype == 'object':
        categorical_columns.append(col)
print(categorical_columns)

#droping irrelevant features
dataset = df.copy()
dataset.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
dataset.dropna(inplace=True,how='any',axis=0)
dataset.info()

#one hot encode categorical fatures
dummies = pd.get_dummies(df[['Sex','Embarked']], dtype='float64',drop_first=True)
dummies.head()
dataset = pd.concat([dataset,dummies],axis=1)
dataset.dropna(inplace=True,how='any',axis=0)
dataset.describe()

#visualize generally to see correlation between features
plt.figure(figsize=(10,7))
sns.pairplot(data=dataset,kind='scatter',palette='magma',hue='Survived')
plt.show()
#or we can see it with corr()
dataset.corr()
#or we can use seaborn's heatmap
sns.heatmap(dataset.corr(),annot=True,fmt=".2f")
plt.show()

#divide dataset into features and labels
X = dataset.drop('Survived', axis=1) #features
y = dataset['Survived']              #target variable

X.drop(['Sex','Embarked'],inplace=True,axis=1)
X.info()
y.info()


#split data into train and test
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
from sklearn.preprocessing import StandardScaler
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#compare data distribution before and after standardization
sns.boxplot(X_train)
plt.show()
sns.boxplot(X_train_scaled)
plt.show()

# build decision tree model
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train_scaled,y_train)
tree_model.score(X_test_scaled,y_test)


#plot the model graph
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(15, 10))
plot_tree(tree_model, filled=True)
plt.gcf().subplots_adjust(left=0.3)
plt.show()

#build svm model
from sklearn.svm import SVC
svm_model = SVC(kernel='poly',gamma='auto',max_iter=10000)
svm_model.fit(X_train_scaled,y_train)
svm_model.score(X_test_scaled,y_test)

