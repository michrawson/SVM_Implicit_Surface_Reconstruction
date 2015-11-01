from sklearn.datasets import make_regression
from sklearn.cross_validation import train_test_split
from sklearn.kernel_ridge import KernelRidge
import sys
import numpy as np

def kernel(a,b):
    return np.dot(a,b)

assert kernel([1,1],[1,-1]) == 0

def kernel_ridge_regression(X_train,y_train, Lambda):
    y_train = np.matrix(y_train).transpose()
    K = np.matrix(np.zeros( (len(X_train), len(X_train)) ))
    for i in range(0, len(X_train)):
        for j in range(0, len(X_train)):
            K[ (i,j) ] = kernel(X_train[i], X_train[j])
    alpha = np.linalg.inv( K + (Lambda*np.identity(len(X_train))) )* y_train
    alpha = np.squeeze(np.asarray(alpha))
    def f(x):
        sum = 0.
        for i in range(0,len(X_train)):
            sum += alpha[i] * kernel(X_train[i],x)
        return sum
    return f

def score(f, X_test, y_test):
    error = 0.
    for i in range(0, len(X_test)):
        prediction = f(X_test[i])
        if isinstance(prediction,np.ndarray):
            prediction = prediction[0]
        error += pow((prediction - y_test[i]),2)
    return error/len(X_test)

# Make up data
X, y, true_coefficient = make_regression(n_samples=80, n_features=30,
                                         n_informative=20, noise=10, coef=True,
                                         random_state=20140210)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)

# Run Scikit Kernel Ridge Regression
clf = KernelRidge()
clf.fit(X_train,y_train)
print 'SCIKIT: mean square test error:', score( clf.predict, X_test, y_test)

# Run this implementation
f = kernel_ridge_regression(X_train,y_train,1)
score_val = score(f, X_test, y_test)
print 'Custom: mean square test error:', score_val
