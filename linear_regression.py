import matplotlib.pyplot as plt
import numpy as  np 
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
#%matplotlib inline used in notebook

diabetes = datasets.load_diabetes()
# diabetes
diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_y = diabetes.target
X_train,X_test,y_train,y_test = train_test_split(diabetes_X,diabetes_y,train_size = 0.80, random_state = 0)
linR = linear_model.LinearRegression()
linR.fit(X_train,y_train)
#print coefficent
print('Coefficients: \n', linR.coef_)
# mean squired error
print('mean squired error is %0.3f'%(np.mean((linR.predict(X_test)-y_test)**2)))
#variance, 0 is best
print('variance is %0.3f'%(linR.score(X_test,y_test)))

# print(X_test)
# print(y_test)
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test,linR.predict(X_test),color='blue',linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()