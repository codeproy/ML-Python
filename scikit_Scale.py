from sklearn import preprocessing
import numpy as np
X_train = np.array([[1,2,3],[10,20,3],[15,-2,10]])
X_scaled = preprocessing.scale(X_train)
print(X_scaled)
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))
