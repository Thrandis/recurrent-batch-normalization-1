import numpy as np

n = 10000000

def sigm(x):
    return 1. / (1 + np.exp(-x))

def normsigm(x):
    y = 1. / (1 + np.exp(-x))
    return (y - 0.5) / np.sqrt(0.043)

def normtanh(x):
    y = np.tanh(x)
    return y / np.sqrt(0.394)

print 'Base'
x = np.random.randn(n)
print np.mean(x), np.var(x)

#print 'ReLU'
#y = np.maximum(0, x)
#print np.mean(y), np.var(y)
#
#print 'ReLU gamma'
#y = np.maximum(0, 1./1.21 * x)
#print np.mean(y), np.var(y)
#
#print 'Norm ReLU not compensated for gamma'
#y = np.maximum(0, 1./1.21 * x)
#y -= 0.399
#y /= np.sqrt(0.3409)
#print np.mean(y), np.var(y)
#
#print 'Norm ReLU gamma'
#y = np.maximum(0, 1./1.21 * x)
#y -= 0.3298
#y /= np.sqrt(0.2328)
#print np.mean(y), np.var(y)

#print 'Sigmoid'
#y = sigm(x)
#print np.mean(y), np.var(y)
#
#print 'Norm Sigmoid'
#y = normsigm(x)
#print np.mean(y), np.var(y)
#
print 'Tanh'
y = np.tanh(x)
print np.mean(y), np.var(y)

print 'Tanh gamma 0.1'
y = np.tanh(0.1*x)
print np.mean(y), np.var(y)

print 'Tanh gamma 0.8'
y = np.tanh(0.8*x)
print np.mean(y), np.var(y)

#print 'Norm Tanh'
#y = normtanh(x)
#print np.mean(y), np.var(y)
#
#print 'sigm(x) * z'
#y = sigm(x) * np.random.randn(n) 
#print np.mean(y), np.var(y)
#
#print 'normsigm(x) * z'
#y = normsigm(x) * np.random.randn(n) 
#print np.mean(y), np.var(y)
#
#print 'tanh(x) * z'
#y = np.tanh(x) * np.random.randn(n) 
#print np.mean(y), np.var(y)
#
#print 'normtanh(x) * z'
#y = normtanh(x) * np.random.randn(n) 
#print np.mean(y), np.var(y)
#del x, y
#
#print 'Sum of 2 variables'
#z = np.random.randn(n) + np.random.randn(n)
#print np.mean(z), np.var(z)
#
#print 'Norm Sum of 2 variables'
#z = (np.random.randn(n) + np.random.randn(n)) / np.sqrt(2.)
#print np.mean(z), np.var(z)
#del z
#
#
#
#print 'LSTM (c, h)'
#c0 = np.random.randn(n)
#f = np.random.randn(n) + np.random.randn(n)
#i = np.random.randn(n) + np.random.randn(n)
#o = np.random.randn(n) + np.random.randn(n)
#g = np.random.randn(n) + np.random.randn(n)
#c = sigm(f) * c0 + sigm(i) * np.tanh(g)
#h = sigm(o) * np.tanh(c)
#print np.mean(c), np.var(c)
#print np.mean(h), np.var(h)
#
#print 'Norm LSTM (c, h)'
#c0 = np.random.randn(n)
#f = (np.random.randn(n) + np.random.randn(n)) / np.sqrt(2.)
#i = (np.random.randn(n) + np.random.randn(n)) / np.sqrt(2.)
#o = (np.random.randn(n) + np.random.randn(n)) / np.sqrt(2.)
#g = (np.random.randn(n) + np.random.randn(n)) / np.sqrt(2.)
#c = (normsigm(f) * c0 + normsigm(i) * normtanh(g)) / np.sqrt(2.)
#h = normsigm(o) * normtanh(c)
#print np.mean(c), np.var(c)
#print np.mean(h), np.var(h)


#x = np.random.randn(n, 1)
#gamma = np.linspace(0, 2, 1000).reshape(1000, 1)
#o = np.tanh(np.dot(x, gamma.T))
#
#m = o.mean(axis=0)
#v = o.var(axis=0)
#print m.shape
#del o 
#
#import matplotlib.pylab as plt
#plt.figure()
#plt.title('mean')
#plt.plot(gamma, m)
#
#plt.figure()
#plt.title('var')
#plt.plot(gamma, v)
#
#plt.show()
