# %% [code] {"execution":{"iopub.execute_input":"2023-04-14T13:38:37.079630Z","iopub.status.busy":"2023-04-14T13:38:37.079071Z","iopub.status.idle":"2023-04-14T13:38:37.092504Z","shell.execute_reply":"2023-04-14T13:38:37.091213Z","shell.execute_reply.started":"2023-04-14T13:38:37.079586Z"}}
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

# %% [code] {"execution":{"iopub.execute_input":"2023-04-14T13:39:18.785937Z","iopub.status.busy":"2023-04-14T13:39:18.785486Z","iopub.status.idle":"2023-04-14T13:39:18.899530Z","shell.execute_reply":"2023-04-14T13:39:18.898185Z","shell.execute_reply.started":"2023-04-14T13:39:18.785883Z"}}
train_data = pd.read_csv("data/train.csv")
train_data.head()

# %% [code] {"execution":{"iopub.execute_input":"2023-04-14T13:39:32.899999Z","iopub.status.busy":"2023-04-14T13:39:32.899520Z","iopub.status.idle":"2023-04-14T13:39:33.037688Z","shell.execute_reply":"2023-04-14T13:39:33.036286Z","shell.execute_reply.started":"2023-04-14T13:39:32.899958Z"}}
test_data = pd.read_csv("data/test.csv")
test_data.head()

# %% [code] {"execution":{"iopub.execute_input":"2023-04-14T13:39:48.007432Z","iopub.status.busy":"2023-04-14T13:39:48.006994Z","iopub.status.idle":"2023-04-14T13:39:48.015044Z","shell.execute_reply":"2023-04-14T13:39:48.013560Z","shell.execute_reply.started":"2023-04-14T13:39:48.007392Z"}}
print("Numerical Variables")
numerical_variables = train_data._get_numeric_data().columns
for col in numerical_variables:
    print(col)

# %% [code]
"""plt.figure(figsize=(10,21))
i = 0
for ncol in numerical_variables:
    i += 1
    plt.subplot(2, 2, i)
    train_data[ncol].value_counts().plot(kind='bar', title=ncol)
plt.tight_layout()"""

# %% [code] {"execution":{"iopub.execute_input":"2023-04-14T13:52:28.078657Z","iopub.status.busy":"2023-04-14T13:52:28.078262Z","iopub.status.idle":"2023-04-14T13:52:28.087707Z","shell.execute_reply":"2023-04-14T13:52:28.086364Z","shell.execute_reply.started":"2023-04-14T13:52:28.078622Z"}}
X = train_data.drop("VERDICT", axis=1).values
y = train_data["VERDICT"].values

# %% [code] {"execution":{"iopub.execute_input":"2023-04-14T13:52:30.306703Z","iopub.status.busy":"2023-04-14T13:52:30.306286Z","iopub.status.idle":"2023-04-14T13:52:30.319146Z","shell.execute_reply":"2023-04-14T13:52:30.317896Z","shell.execute_reply.started":"2023-04-14T13:52:30.306666Z"}}
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

X_train.shape,y_train.shape, X_test.shape, y_test.shape

# %% [code] {"execution":{"iopub.execute_input":"2023-04-14T13:52:52.139129Z","iopub.status.busy":"2023-04-14T13:52:52.138694Z","iopub.status.idle":"2023-04-14T13:52:57.186908Z","shell.execute_reply":"2023-04-14T13:52:57.185423Z","shell.execute_reply.started":"2023-04-14T13:52:52.139089Z"}}
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
rf_pred_score = rf.score(X_test,y_test)

# %% [code] {"execution":{"iopub.execute_input":"2023-04-14T13:53:24.704812Z","iopub.status.busy":"2023-04-14T13:53:24.703744Z","iopub.status.idle":"2023-04-14T13:53:28.671151Z","shell.execute_reply":"2023-04-14T13:53:28.669504Z","shell.execute_reply.started":"2023-04-14T13:53:24.704753Z"}}
gb = GradientBoostingClassifier()
gb.fit(X_train,y_train)
gb_pred_score = gb.score(X_test,y_test)

# %% [code] {"execution":{"iopub.execute_input":"2023-04-14T13:53:38.784664Z","iopub.status.busy":"2023-04-14T13:53:38.784242Z","iopub.status.idle":"2023-04-14T13:53:49.721596Z","shell.execute_reply":"2023-04-14T13:53:49.720614Z","shell.execute_reply.started":"2023-04-14T13:53:38.784626Z"}}
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)
svc_pred_score = svc.score(X_test,y_test)

# %% [code] {"execution":{"iopub.execute_input":"2023-04-14T13:53:53.139608Z","iopub.status.busy":"2023-04-14T13:53:53.139189Z","iopub.status.idle":"2023-04-14T13:53:53.426649Z","shell.execute_reply":"2023-04-14T13:53:53.424898Z","shell.execute_reply.started":"2023-04-14T13:53:53.139569Z"}}
lg = LogisticRegression()
lg.fit(X_train,y_train)
lg_pred_score = lg.score(X_test,y_test)

# %% [code] {"execution":{"iopub.execute_input":"2023-04-14T13:54:06.096221Z","iopub.status.busy":"2023-04-14T13:54:06.095106Z","iopub.status.idle":"2023-04-14T13:54:06.110536Z","shell.execute_reply":"2023-04-14T13:54:06.108995Z","shell.execute_reply.started":"2023-04-14T13:54:06.096163Z"}}
df = pd.DataFrame(dict(model=['Logistic Regression', 
                              'Random Forest', 
                              'Gradient Boosting',
                              'SVM'],accuracy=[lg_pred_score, rf_pred_score, 
                                               gb_pred_score, svc_pred_score]))
df

# %% [code] {"execution":{"iopub.execute_input":"2023-04-14T13:54:29.900728Z","iopub.status.busy":"2023-04-14T13:54:29.900332Z","iopub.status.idle":"2023-04-14T13:54:30.172267Z","shell.execute_reply":"2023-04-14T13:54:30.171137Z","shell.execute_reply.started":"2023-04-14T13:54:29.900693Z"}}
df.plot(kind='bar',x='model',y='accuracy',title='Model Accuracy',legend=False,
        color=['#1F77B4', '#FF7F0E', '#2CA02C'])
plt.ylim(0.5,1)

# %% [code] {"execution":{"iopub.execute_input":"2023-04-14T13:58:33.181696Z","iopub.status.busy":"2023-04-14T13:58:33.181147Z","iopub.status.idle":"2023-04-14T13:58:33.202975Z","shell.execute_reply":"2023-04-14T13:58:33.201532Z","shell.execute_reply.started":"2023-04-14T13:58:33.181644Z"}}
test_data.info()

# %% [code] {"execution":{"iopub.execute_input":"2023-04-14T13:59:09.636145Z","iopub.status.busy":"2023-04-14T13:59:09.634953Z","iopub.status.idle":"2023-04-14T13:59:09.645106Z","shell.execute_reply":"2023-04-14T13:59:09.643989Z","shell.execute_reply.started":"2023-04-14T13:59:09.636098Z"}}
X_train = train_data.drop("VERDICT", axis=1)
y_train = train_data['VERDICT']
X_test = test_data.drop("Id", axis=1).copy()

# %% [code] {"execution":{"iopub.execute_input":"2023-04-14T13:59:12.693780Z","iopub.status.busy":"2023-04-14T13:59:12.692718Z","iopub.status.idle":"2023-04-14T13:59:19.887465Z","shell.execute_reply":"2023-04-14T13:59:19.886283Z","shell.execute_reply.started":"2023-04-14T13:59:12.693716Z"}}
model = RandomForestClassifier()
model.fit(X_train,y_train)
predictions = model.predict(X_test)

# %% [code] {"execution":{"iopub.execute_input":"2023-04-14T14:03:31.205548Z","iopub.status.busy":"2023-04-14T14:03:31.205128Z","iopub.status.idle":"2023-04-14T14:03:31.275853Z","shell.execute_reply":"2023-04-14T14:03:31.274348Z","shell.execute_reply.started":"2023-04-14T14:03:31.205512Z"}}
output = pd.DataFrame({'Id':test_data['Id'],'VERDIT':predictions})
output.to_csv('predictions.csv', index=False)
print("Your submission was successfully saved!")

# %% [code] {"execution":{"iopub.execute_input":"2023-04-14T14:03:44.516614Z","iopub.status.busy":"2023-04-14T14:03:44.516183Z","iopub.status.idle":"2023-04-14T14:03:44.539477Z","shell.execute_reply":"2023-04-14T14:03:44.538169Z","shell.execute_reply.started":"2023-04-14T14:03:44.516575Z"}}
sub = pd.read_csv("predictions.csv")
sub.head(20)