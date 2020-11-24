import numpy as np
import torch
from torch import nn

# From cs231n
def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    
    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.
    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.
    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
        - mode: 'train' or 'test'; required
        - eps: Constant for numeric stability
        - momentum: Constant for running mean / variance.
        - running_mean: Array of shape (D,) giving running mean of features
        - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(x, axis = 0)
        sample_var = np.var(x , axis = 0)
        x_hat = (x - sample_mean) / (np.sqrt(sample_var  + eps))
        out = gamma * x_hat + beta
        cache = (gamma, x, sample_mean, sample_var, eps, x_hat)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
    elif mode == 'test':
        scale = gamma / (np.sqrt(running_var  + eps))
        out = x * scale + (beta - running_mean * scale)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


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