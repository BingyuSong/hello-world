import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
%matplotlib inline

#  NPPPNNPNNPNN
fig = plt.figure()
x = [0,1/7,1/7,1/7,1/7,2/7,3/7,3/7,4/7,5/7,5/7,6/7,7/7]# false_positive_rate
y = [0,0,1/5,2/5,3/5,3/5,3/5,4/5,4/5,4/5,5/5,5/5,5/5]# true_positive_rate 
# print(len(x))
# print(len(y))
# This is the ROC curve
fig, ax = plt.subplots()
ax.plot(x,y)
ax.plot(x,y, 'ro')
start, end = ax.get_xlim()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
ax.xaxis.set_ticks(np.arange(start, end, 1/7))
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))
ax.yaxis.set_ticks(np.arange(start, end, 1/5))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))

ax.grid(which='both')                                                            

# or if you want differnet settings for the grids:                               
ax.grid(which='minor', alpha=0.2)                                                
ax.grid(which='major', alpha=0.5)       
plt.title('Roc_quiz')
plt.savefig('roc_picture')
plt.show() 

 # This is the AUC
auc = np.trapz(y,x)
