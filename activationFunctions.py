import numpy as np

def sigmoid(x, prime):
    if not prime:
        ret =  np.array(1.0/(1.0+np.exp(-x)))
        return ret
    else:
        sig = sigmoid(x, False)
        ret = np.array(sig*(1.0-sig))
        return ret

def relu(x, prime):
    ret = np.zeros(len(x)).reshape(len(x),1)
    if not prime:
        return np.maximum(x,0.0)
    else:
        ret[x<0.0] = 0.0
        ret[x>=0.0] = 1.0
        return ret

def tanh(x, prime):
    if not prime:
        ret = np.array(np.tanh(x))
        return ret
    else:
        ret - np.array(1.0-(np.tanh(x))^2)
        return ret

#terminar a leaky relu para array !!!!!!!!!!!!!!!!!!!!!!!
def leaky_relu(x, prime):
    if not prime:
        if x < 0:
            return 0.01*x
        else:
            return x
    else:
        if x < 0:
            return 0.01
        else:
            return 1
