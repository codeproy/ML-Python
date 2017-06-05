import os  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sigmoid import sigmoid


path = os.getcwd() + r'\ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam1','Exam2', 'Admitted'])
pos = data[data['Admitted'].isin([1])]
neg = data[data['Admitted'].isin([0])]
#print(data.head(10))
#print(pos.head(2))
#print(neg.head(2))
fig, ax = plt.subplots(figsize=(12,8))
#ax.plot(x, f, 'r', label='Prediction')
#ax.scatter(pos.Exam1, pos.Exam2, label='Admitted',s=50, c='b', marker='P')
#ax.scatter(neg.Exam1, neg.Exam2, label='Non Admit',s=50, c='y', marker='o')
#ax.legend(loc=2)
#ax.set_xlabel('Exam1')
#ax.set_ylabel('Exam2')
#ax.set_title('School Admission')
#plt.show()


def costFunction(theta, X ,y):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    J = 0
    m = len(X)
    pred = sigmoid(X * theta.T)
    J = 1/m * ( np.sum(-np.multiply(y ,np.log(pred)) - np.multiply((1-y), np.log(1- pred))))
#    G = 1 /m * (np.multiply(X, (pred-y)))
    return J 

def Gradient(theta, X ,y):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    parameters = int(theta.ravel().shape[1])
    G = np.zeros(parameters)
#    G = 0
    m = len(X)

    for i in range(parameters):
        pred = sigmoid(X * theta.T)
#    J = 1/m * ( np.sum(-np.multiply(y ,np.log(pred)) - np.multiply((1-y), np.log(1- pred))))
        G[i] = 1 /m * np.sum((np.multiply(X[:,i], (pred-y))))
   
    return G
    
    

def gradient(theta, X, y):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)


    print('sh',grad.shape)
    return grad

data.insert(0,'Ones',1)
cols = data.shape[1]
#print('cols',cols)
X = data.iloc[:,0:cols - 1]
y = data.iloc[:,cols-1:cols]
X = np.array(X.values)
y = np.array(y.values)
theta = np.array(np.zeros(cols-1))
j = costFunction(theta,X,y)
print('Cost with theta zero', j)
test_theta = np.array([-24,0.2, 0.2])
j = costFunction(test_theta,X,y)
print('Cost with test theta', j)

theta = np.array(np.zeros(cols-1))

#Result = op.minimize(fun = costFunction,x0 = initial_theta, args = (X, y),method = 'TNC',jac = gradient)
result = opt.fmin_tnc(func=costFunction, x0=theta, fprime=Gradient, args=(X, y))  
optimal_theta = result[0]

j = costFunction(optimal_theta,X,y)
print('Cost with optimal theta', j)

print ('completed')
#a = np.arange(-10,10,step=1)
#m = sigmoid(a)
#fig, ax = plt.subplots(figsize=(12,8))
#ax.plot(a,m,'r')
#plt.show()
#y = np.matrix(y.values)
#print ('X ' , X[0:3,:])







