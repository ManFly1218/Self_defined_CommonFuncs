import numpy as np

def mean_square_error(y,t):
    n=y.shape[0]
    MSE=0.5*np.sum((y-t)**2)/n
    return MSE

def cross_entropy_error(y,t):
    delta=1e-7
    return -np.sum(t*np.log(y+delta))

def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

def sigmoid(x):
    return 1/(1+np.exp(-x))

def numerical_gradient(f,x):
    #x = x.astype(float)#convert the input matrix as float type
    grad=np.zeros_like(x)#
    h=1e-4
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    for i in it:
        idx=it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx]= (fxh1-fxh2) / (2*h)
        
        x[idx]=tmp_val
        
    return grad