import torch
import torch.nn as nn
from torch.nn import Parameter, init

class MyBatchNorm(nn.Module):
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MyBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.uniform_(self.weight)
            init.zeros_(self.bias)
        
    def forward(self, input):        
        input_size = input.size()
        input = input.transpose(1,0)
        input = input.view(input.size(0), -1)

        if self.training:
            mean = input.mean(dim=1)
            var = torch.var(input,dim=1, unbiased=True)
            self.running_mean[:] = (1. - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var[:] = (1. - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        input = input - mean.view(-1,1)
        input = input / (torch.sqrt(var+self.eps).view(-1,1))
       
        input = self.weight.view(-1, 1) * input + self.bias.view(-1, 1)
        input = input.transpose(1,0)
        input = input.view(*input_size)
        return input

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(MyBatchNorm, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

def test_batch_norm():
    momentum = 1.0
    torch.manual_seed(1234)

    batch_norm = MyBatchNorm(3,eps=1e-5,momentum=momentum,affine=True,track_running_stats=True)
    n1 = batch_norm
    torch.save(n1.state_dict(),'n1.pth')
    
    torch_batch_norm = nn.BatchNorm1d(3,eps=1e-5,momentum=momentum,affine=True,track_running_stats=True)
    n2 = torch_batch_norm
    n2.load_state_dict(torch.load('n1.pth'))
    
    x = torch.FloatTensor([[1,2,3], [3,4,0], [3,3,1]])
    y = torch.FloatTensor([[2], [3], [1]])
    criterion = nn.MSELoss()

    x = x.cuda()
    y = y.cuda()
    batch_norm.cuda()
    torch_batch_norm.cuda()

    print('Switch to eval mode.')
    batch_norm.eval()
    torch_batch_norm.eval()
    out1 = n1(x)
    out2 = n2(x)
    eval1 = (torch.abs(out2-out1).sum().item() < 1e-4)

    print('Swtich to train mode.')
    batch_norm.train()
    torch_batch_norm.train()

    out1 = n1(x)
    out2 = n2(x)
    train2 = (torch.abs(out2-out1).sum().item() < 1e-4)

    print('Switch to eval mode.')
    n1.eval()
    n2.eval()
    print('MyBatchNorm:')
    print('running_mean:',batch_norm.running_mean.cpu().numpy())
    print('running_var:',batch_norm.running_var.cpu().numpy())
    print('weight:',batch_norm.weight.data.cpu().numpy())
    print('bias:',batch_norm.bias.data.cpu().numpy())
    print()
    
    print('TorchBatchNorm:')
    print('running_mean:',torch_batch_norm.running_mean.cpu().numpy())
    print('running_var:',torch_batch_norm.running_var.cpu().numpy())
    print('weight:',torch_batch_norm.weight.data.cpu().numpy())
    print('bias:',torch_batch_norm.bias.data.cpu().numpy())
    out1 = n1(x)
    out2 = n2(x)
    eval3 = (torch.abs(out2-out1).sum().item() < 1e-4)
    print('eval1,train2,eval3:',eval1,train2,eval3)
    assert eval1 and train2 and eval3

if __name__ == '__main__':
    test_batch_norm()