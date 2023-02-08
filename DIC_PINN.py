import torch
import torch.nn as nn
import numpy as np

class FF_PINN(nn.Module):
    def __init__(self, layers, activation_fn = 'ReLU',enforce_BC = lambda x,u: u):
        super(FF_PINN, self).__init__()

        self.layers = nn.ModuleList()
        self.enforce_BC = enforce_BC

        for input_size, output_size in zip(layers, layers[1:]):
            self.layers.append(nn.Linear(input_size, output_size))
        
        if activation_fn == 'ReLU':
            self.activation = nn.ReLU()
        elif activation_fn == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_fn == 'Tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Invalid activation function")
            
    def forward(self, x):
        x_in = x
        for i, linear in enumerate(self.layers):
            x_in = linear(x_in)
            if i != len(self.layers) - 1:
                x_in = self.activation(x_in)

            x_out = self.enforce_BC(x,x_in)
        return x_out
        
    def train(self,num_epochs, optimizer,scheduler, PDE_sampling, PDE_loss):
        # num_epochs is an integer
        # optimizer is a torch optimizer
        # PDE_sampling is a tensor
        # PDE_loss is a function
        # returns a tensor of the solution

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            PDE_loss(PDE_sampling).backward()
            optimizer.step()
            if epoch % 100 == 0:
                print('Epoch: %d, Loss: %.3e' % (epoch, PDE_loss(PDE_sampling).item()))
            scheduler.step()
        

        
def domain_sampling(domain,sampling_strat,num_samples):
    # domain is a list of tuples
    # sampling_strat is a string
    # num_samples is an integer
    # returns a tensor of points

    if sampling_strat == 'uniform':
        # uniform sampling
        x = np.linspace(domain[0][0],domain[0][1],num_samples)
        y = np.linspace(domain[1][0],domain[1][1],num_samples)
        X,Y = np.meshgrid(x,y)
        X = X.reshape(-1,1)
        Y = Y.reshape(-1,1)
        XY = torch.Tensor(np.concatenate((X,Y),axis=1))
        XY.requires_grad = True
        return XY
    
    elif sampling_strat == 'random':
        # random sampling
        x = np.random.uniform(domain[0][0],domain[0][1],num_samples)
        y = np.random.uniform(domain[1][0],domain[1][1],num_samples)
        X,Y = np.meshgrid(x,y)
        X = X.reshape(-1,1)
        Y = Y.reshape(-1,1)
        return torch.Tensor(np.concatenate((X,Y),axis=1))

def lr_scheduler(optimizer, strategy, num_epochs):
    # strategy is a string
    # num_epochs is an integer

    if strategy == 'step':
        # step learning rate decay
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        return scheduler

    elif strategy == 'exp':
        # exponential learning rate decay
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return scheduler

    elif strategy == 'cos':
        # cosine learning rate decay
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
        return scheduler
    
    elif strategy == 'constant':
        # constant learning rate
        return optimizer

    else:
        raise ValueError("Invalid learning rate strategy")