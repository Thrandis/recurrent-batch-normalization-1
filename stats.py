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

print 'Sigmoid'
y = sigm(x)
print np.mean(y), np.var(y)

print 'Norm Sigmoid'
y = normsigm(x)
print np.mean(y), np.var(y)

print 'Tanh'
y = np.tanh(x)
print np.mean(y), np.var(y)

print 'Norm Tanh'
y = normtanh(x)
print np.mean(y), np.var(y)

print 'sigm(x) * z'
y = sigm(x) * np.random.randn(n) 
print np.mean(y), np.var(y)

print 'normsigm(x) * z'
y = normsigm(x) * np.random.randn(n) 
print np.mean(y), np.var(y)

print 'tanh(x) * z'
y = np.tanh(x) * np.random.randn(n) 
print np.mean(y), np.var(y)

print 'normtanh(x) * z'
y = normtanh(x) * np.random.randn(n) 
print np.mean(y), np.var(y)
del x, y

print 'Sum of 2 variables'
z = np.random.randn(n) + np.random.randn(n)
print np.mean(z), np.var(z)

print 'Norm Sum of 2 variables'
z = (np.random.randn(n) + np.random.randn(n)) / np.sqrt(2.)
print np.mean(z), np.var(z)
del z



print 'LSTM (c, h)'
c0 = np.random.randn(n)
f = np.random.randn(n) + np.random.randn(n)
i = np.random.randn(n) + np.random.randn(n)
o = np.random.randn(n) + np.random.randn(n)
g = np.random.randn(n) + np.random.randn(n)
c = sigm(f) * c0 + sigm(i) * np.tanh(g)
h = sigm(o) * np.tanh(c)
print np.mean(c), np.var(c)
print np.mean(h), np.var(h)

print 'Norm LSTM (c, h)'
c0 = np.random.randn(n)
f = (np.random.randn(n) + np.random.randn(n)) / np.sqrt(2.)
i = (np.random.randn(n) + np.random.randn(n)) / np.sqrt(2.)
o = (np.random.randn(n) + np.random.randn(n)) / np.sqrt(2.)
g = (np.random.randn(n) + np.random.randn(n)) / np.sqrt(2.)
c = (normsigm(f) * c0 + normsigm(i) * normtanh(g)) / np.sqrt(2.)
h = normsigm(o) * normtanh(c)
print np.mean(c), np.var(c)
print np.mean(h), np.var(h)
