from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

from sklearn.model_selection import train_test_split

X = data.data
y = data.target

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size = 0.67, random_state = 0)
from sklearn.naive_bayes import GaussianNB
cancer_model=GaussianNB()
cancer_model.fit(x_train,y_train)
cancer_model.score(x_test,y_test)
