
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pathlib

train_data = pd.read_csv("data/train.csv")
train_data.head()

test_data = pd.read_csv("data/test.csv")
test_data.head()

print("Numerical Variables")
numerical_variables = train_data._get_numeric_data().columns
for col in numerical_variables:
    print(col)

"""plt.figure(figsize=(10,21))
i = 0
for ncol in numerical_variables:
    i += 1
    plt.subplot(2, 2, i)
    train_data[ncol].value_counts().plot(kind='bar', title=ncol)
plt.tight_layout()"""

X = train_data.drop("VERDICT", axis=1).values
y = train_data["VERDICT"].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

X_train.shape,y_train.shape, X_test.shape, y_test.shape

rf = RandomForestClassifier()
rf.fit(X_train,y_train)
rf_pred_score = rf.score(X_test,y_test)

gb = GradientBoostingClassifier()
gb.fit(X_train,y_train)
gb_pred_score = gb.score(X_test,y_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)
svc_pred_score = svc.score(X_test,y_test)

lg = LogisticRegression()
lg.fit(X_train,y_train)
lg_pred_score = lg.score(X_test,y_test)

df = pd.DataFrame(dict(model=['Logistic Regression', 
                              'Random Forest', 
                              'Gradient Boosting',
                              'SVM'],accuracy=[lg_pred_score, rf_pred_score, 
                                               gb_pred_score, svc_pred_score]))
df

df.plot(kind='bar',x='model',y='accuracy',title='Model Accuracy',legend=False,
        color=['#1F77B4', '#FF7F0E', '#2CA02C'])
plt.ylim(0.5,1)

test_data.info()

X_train = train_data.drop("VERDICT", axis=1)
y_train = train_data['VERDICT']
X_test = test_data.drop("Id", axis=1).copy()

model = RandomForestClassifier()
model.fit(X_train,y_train)
predictions = model.predict(X_test)

output = pd.DataFrame({'Id':test_data['Id'],'VERDIT':predictions})
output.to_csv('predictions.csv', index=False)
print("Your submission was successfully saved!")

sub = pd.read_csv("predictions.csv")
sub.head(20)