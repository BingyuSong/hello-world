from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import numpy as np

data = load_breast_cancer()
from sklearn.model_selection import train_test_split
X = data.data
y = data.target
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.67, random_state = 3)

clf = LogisticRegression()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import decision_function

Accu_score = accuracy_score(y_test,y_pred)
Prec_score = precision_score(y_test,y_pred)
Reca_score = recall_score(y_test,y_pred)
Roc_Auc_score = roc_auc_score(y_test,clf.decision_function(X_test))

print("accuracy_score=%0.3f\n precision_score=%0.3f\n recall_score=%0.3f\n roc_auc_score=%0.3f\n"%(Accu_score,Prec_score,Reca_score,Roc_Auc_score))

