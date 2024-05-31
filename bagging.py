import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# demonstrate the bagging technique of ensemble methods

#a) generate a random data set with 20 samples in 2 classes
rng = np.random.default_rng(seed=12345)
X = rng.random((20, 2))
y = rng.integers(0, 2, size=20)

print(X)
print(y)

#b) generate 10 training data sets each of size 20 by sampling with 
#   repetition. Give the output as a 10x20 matrix

Trains=np.random.randint(20, size=(10,20))

#Each integer 0-19 represents a row of the data set 
#(element 9 will access row 9 of X and y))

print(Trains)

# d) Train 10 classifier models on the 10 data sets
LogReg1=LogisticRegression(penalty=None,C=1,random_state=0,solver='saga',max_iter=10000)
LogReg1.fit(X,y)
LogReg1.predict(X)
