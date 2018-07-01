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
        ret[x<0] = 0.0
        ret[x>=0] = 1.0
        return ret

def tanh(x, prime):
    if not prime:
        ret = np.array(np.tanh(x))
        return ret
    else:
        ret = np.array(1.0 - (np.tanh(x))**2)
        return ret

#terminar a leaky relu para array !!!!!!!!!!!!!!!!!!!!!!!
def leaky_relu(x, prime):
    neg = np.argwhere(x < 0)
    if not prime:
        x[neg[:,0]] = 0.01 * x[neg[:,0]]
    else:
        x[:] = 1
        x[neg[:,0]] = 0.01

    return x
