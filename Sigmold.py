
# coding: utf-8

# In[2]:

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
import numpy as np
data = load_breast_cancer()
X = data.data
y = data.target
model = MLPClassifier(random_state=9)
accuracy = cross_val_score(model, X, y,scoring='accuracy', cv = 5)
print('mean of accuracy is % 0.3f\n'%(np.mean(accuracy)))


# In[ ]:

import numpy as np


# # Binary Sigmold

# In[34]:

def Binary_Sigmold(k):
    return 1/(1+np.exp(-k))
#Binary_S
def Bipolar_Sigmold(k):
    return  (2/(1+np.exp(-k)))+1
def Tangent(k):
    return(np.exp(k)-np.exp(-k))/(np.exp(k)+np.exp(-k))
def error_regression(t, y):
    return (1/2)*(t-y)**2
def error_classical(t,y):
    return -(1-t)*np.log(1-y)-t*np.log(y)


# 

# In[52]:

import numpy as np
w = [2,2,1,-3]
x = [1,2,-5,1]
k=0
for i in range(len(w)):
    k += w[i]*x[i]
s=Binary_Sigmold(k)
print('%0.3f'%s)


# In[47]:

import numpy as np
w1=[2,-1,2]
v01=[2,1,1,-2]
v02 = [-2,1,-1,-2]
x=[1,2,-1,3]
k01=0
k02=0
for j in range(0,len(v01)):
    k01 += v01[j]*x[j]
    k02 += v02[j]*x[j]
print(k01)
print(k02)
z01 = Tangent(k01)
z02 = Tangent(k02)
print(z01)
print(z02)
#print(z)
k2 = w1[0]+z01*w1[1]+z02*w1[2]
y = Binary_Sigmold(k2)
#soso = (1-y)*(1-z**2) 
print(y)


# In[46]:

import numpy as np
x=[1,2,-1,3]
v=[2,1,1,-2]
w=[2,-1]
k=0
for i in range(len(x)):
    k+=x[i]*v[i]
z=Tangent(k)
m = 2-z
y = Binary_Sigmold(m)
print(y)


# In[51]:

import numpy as np
w0=[0,-1]
v0=[1,-2]
x=[1,1]
k1=-1
z=Tangent(k1)
m = -z
y=Binary_Sigmold(m)
k=(1-y)*(1-z**2)
print(k)


# In[ ]:



