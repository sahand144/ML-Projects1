#essential libraries
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

#importing dataset
dataset_dir = r"D:\datasets\Lgistic regression\DiabetesDataAnslysis.csv"
df = pd.read_csv(dataset_dir)

#EDA part
df.head()
df.info()

#drop rows with null values
df.dropna(axis=0,inplace=True,how='any')

df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True,cmap="cool")
plt.title("Correlation Heat Map")
plt.show()

#divide data into features and labels
X = df.iloc[:,:-1]   #features or independent variable
y = df.iloc[:,-1]    #labels or dependent variable
print("\nFeatures: \n", X.columns)
print("\nLabels: \n ", y.name)

#X samples
X.sample(10)

#split data into train and test partition
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

#standardize our features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#create a model object from LogisticRegression class of Sklearn library
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
#Training the logistic regression model on training set
logreg.fit(X_train_scaled,y_train)
#model score:
print ("\nModel Score : ", logreg.score(X_test_scaled,y_test))
#predicting labels for the test set
y_pred = logreg.predict(X_test_scaled)
#Evaluating the performance of the algorithm
from sklearn.metrics import classification_report,accuracy_score
print("\nLogistic Regression Model Performance Metrics:\n")
print(classification_report(y_test,y_pred))
print("Accuracy Score : ", accuracy_score(y_test,y_pred))
#plotting ROC curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve
fpr, tpr, thresholds = roc_curve(y_test,y_pred)
roc_auc = roc_auc_score(y_test,y_pred)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw)
plt.show()

