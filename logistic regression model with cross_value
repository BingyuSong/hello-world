from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

data = load_breast_cancer()
X = data.data
y = data.target
model = LogisticRegression()
accuracy = cross_val_score(model, X, y,scoring='accuracy', cv = 8)
precision = cross_val_score(model, X, y,scoring='precision', cv = 8)
recall = cross_val_score(model, X, y,scoring='recall', cv = 8)
ROC = cross_val_score(model, X, y,scoring='roc_auc', cv = 8)
print('mean of accuracy is % 0.3f\n mean of precision is % 0.3f\n mean of recall is % 0.3f\n mean of ROC is % 0.3f\n'%(np.mean(accuracy),np.mean(precision),np.mean(recall),np.mean(ROC)))
