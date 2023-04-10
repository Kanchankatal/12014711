# -*- coding: utf-8 -*-

# -- Sheet --

import pandas as pd 
sf=pd.read_csv('carbon_nanotubes (1).csv')

sf.head()

sf.shape

sf.isna().sum()

from sklearn.impute import SimpleImputer
si=SimpleImputer(strategy='most_frequent')
sf_im=si.fit_transform(sf)

pd.DataFrame(sf_im).isna().sum()

cols=[0,1,2,3,4,5,6]


sf_im=pd.DataFrame(sf_im)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in cols:
  sf_im[i]=le.fit_transform(sf_im[i])
sf_im.head()

target=sf_im[0]

target.shape

data=sf_im.drop(columns=[0])

data.shape

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(data, target, test_size=0.3, 
                                                  random_state=2)

x_train.shape

x_train.describe()

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train_std=sc.fit_transform(x_train)

pd.DataFrame(x_train_std).describe()



from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier

p=Perceptron(penalty='l1')

p.fit(x_train,y_train)

test_pred=p.predict(x_test)
train_pred=p.predict(x_train)

print("training accuracy: ",accuracy_score(train_pred, y_train))

print("Testing accuracy: ",accuracy_score(test_pred,y_test))

svc1=SVC(kernel='rbf')

svc1.fit(x_train,y_train)

svc_train_pred=svc1.predict(x_train)
svc_test_pred=svc1.predict(x_test)

print("training accuracy SVC: ", accuracy_score(y_train,svc_train_pred))

print("Testing accuracy SVC: ", accuracy_score(y_test, svc_test_pred))

sf1=DecisionTreeClassifier(criterion='entropy', max_depth=6)

sf1.fit(x_train, y_train)

sf1_train_pred=sf1.predict(x_train)
sf1_test_pred=sf1.predict(x_test)

print("Training accuracy Entropy: ", accuracy_score(y_train, sf1_train_pred))

print("testing accuracy entropy: ", accuracy_score(y_test, sf1_test_pred))

plot_tree(sf1)

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train, y_train)

knn_train_pred=knn.predict(x_train)
knn_test_pred=knn.predict(x_test)

print(" score: ", accuracy_score(knn_train_pred, y_train))

print("Testing accuracy: ", accuracy_score(knn_test_pred, y_test))

from sklearn.model_selection import GridSearchCV
fom sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 500],
    'max_depth': [None, 10, 20],
    'max_features': ['auto', 'sqrt']
}

rf = RandomForestClassifier()

gs=GridSearchCV(estimator=rf, param_grid=param, scoring='accuracy', cv=5)
gs.fit(data, target)

print("Best HyperParameters: ", gs.best_params_)
print("Best Score: ", gs.best_score_)

