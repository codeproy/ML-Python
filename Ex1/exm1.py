import os  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




path = os.getcwd() + r'\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit']) 


#print(data.head())
#print(data.describe())
#data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
#plt.show()


def computeCost(X,y,theta):

    J = 0
    m = len(X)
    pred = X * theta.T
    sqrer = np.power((pred - y),2)
    J = 1 /(2 * m) * np.sum(sqrer)
    return J

def gradientDescent (X, y, theta,alpha, num_iters):
    m = len(X)
    temp = np.matrix(np.zeros(theta.shape))
    Jhist = np.zeros((num_iters,1))

    for i in range(num_iters):
        pdiff = (X * theta.T - y).T * X
        temp[0,0] = temp[0,0] - ((alpha * (1/m)) * np.sum(pdiff[:,0]))
        temp[0,1] = temp[0,1] - (alpha * (1/m) * np.sum(pdiff[:,1]))
        theta = temp
        Jhist[i] = computeCost(X,y,theta)

    return theta
    
def gradientDescent1(X, y, theta, alpha, iters):  
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta
        
    

data.insert(0,'Ones',1)
cols = data.shape[1]
X = data.iloc[:,0:cols - 1]
y = data.iloc[:,cols-1:cols]
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))
alpha = 0.01
iterations = 1000
#print (X.shape)
#print(y.shape)
#print(theta.shape)

print ('Initial cost ', computeCost(X,y,theta))
theta1 = gradientDescent(X, y, theta,alpha, iterations)
print("Theta Optimized :", theta1)
print ('Optmized cost ', computeCost(X,y,theta1))
predict1 = np.dot(np.array([1, 3.5]) , theta1.T)
print ('Predicted Profit ' ,predict1.item((0,0)))

# Visualization
N = 100
x = np.c_[np.ones((N,1)) ,(np.linspace(data.Population.min(),data.Population.max(),N))]
f = np.dot(x ,theta1.T)
#print(f)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()



