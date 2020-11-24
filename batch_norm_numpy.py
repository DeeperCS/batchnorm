import numpy as np
import torch
from torch import nn

class MyBN:
    def __init__(self, momentum, eps, num_features):
        self._running_mean = 0
        self._running_var = 1
        self._momentum = momentum
        self._eps = eps
        self._beta = np.zeros(shape=(num_features,))
        self._gamma = np.ones(shape=(num_features,))
        
    def batch_norm(self, x):
        # mini-batch mean and variance 
        x_mean = x.mean(axis=0)
        x_var = x.var(axis=0)
        # increamentally update the running mean and variance of the whole dataset
        self._running_mean = (1-self._momentum)*x_mean + self._momentum*self._running_mean
        self._running_var = (1-self._momentum)*x_var + self._momentum*self._running_var
        # normalize the current batch of data
        x_hat =(x - x_mean) / np.sqrt(x_var + self._eps)
        # scale and shift (identify transform)
        y = self._gamma * x_hat + self._beta
        return y
    
# Test data
data = np.array([[1, 2],
               [3, 4],
               [1, 4]]).astype(np.float32)

# Pytorch batch norm
bn_torch = nn.BatchNorm1d(num_features=2)
my_bn = MyBN(momentum=0.1, eps=1e-05, num_features=2)


print("Data")
print("data mean:", data.mean(axis=0))
print("data var:", data.var(axis=0))
print() 

data_torch = torch.from_numpy(data)
bn_output_torch = bn_torch(data_torch)
print("Pytorch batch norm")

print("Pytorch batch norm - batch mean var")
print("running_mean:", bn_torch.running_mean)
print("running_var:", bn_torch.running_var)

print(bn_output_torch.detach().numpy())
print()


# my_bn._gamma = bn_torch.weight.detach().numpy()
# my_bn._beta = bn_torch.bias.detach().numpy()
bn_output_numpy = my_bn.batch_norm(data)
print("Numpy batch norm")

print("Pytorch batch norm - batch mean var")
print("running_mean:", my_bn._running_mean)
print("running_var:", my_bn._running_var)

print(bn_output_numpy)