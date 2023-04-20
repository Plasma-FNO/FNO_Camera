#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu July 13 2022


@author: vgopakum

Plots for the RBA Camera data modelled using FNO 
"""

# %%
configuration = {"Case": 'RBA Camera',
                #  "Type": 'Elman RNN',
                 "Pipeline": 'Sequential',
                 "Calibration": 'Calcam',
                 "Epochs": 500,
                 "Batch Size": 20,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 50,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GeLU',
                 "Normalisation Strategy": 'Min-Max',
                 "T_in": 10, 
                 "T_out": 60,
                 "Step": 10,
                 "Modes":8,
                 "Width": 32,
                #  "Hidden Size":16,
                #  "Cells": 1,
                 "Variables": 1,
                 "Resolution":1, 
                 "Noise":0.0}

from simvue import Run
run = Run(mode='disabled')
# run.init(folder="/FNO_Camera", tags=['FRNN', 'Camera', 'rba', 'Forecasting', 'shot-aware', 'linear hidden'], metadata=configuration)


# %%

import numpy as np
from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib import cm 

import operator
from functools import reduce
from functools import partial

import time 
from timeit import default_timer

torch.manual_seed(0)
np.random.seed(0)

import os 
path = file_loc = os.getcwd()
data_loc = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
model_loc = os.getcwd()



#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()


#normalization, rangewise but single value. 
class MinMax_Normalizer(object):
    def __init__(self, x, low=-1.0, high=1.0):
        super(MinMax_Normalizer, self).__init__()
        mymin = torch.min(x)
        mymax = torch.max(x)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()



#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
# %%

#Adding Gaussian Noise to the training dataset
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = torch.FloatTensor([mean])
        self.std = torch.FloatTensor([std])
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).cuda() * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# additive_noise = AddGaussianNoise(0.0, configuration['Noise'])
# additive_noise.cuda()


# %%
################################################################
# Loading Data 
################################################################


data =  np.load(data_loc + '/Data/Cam_Data/Cleaned_Data/rba_30280_30360.npy')
data_2 = np.load(data_loc + '/Data/Cam_Data/rba_fno_data_2.npy')
# data =  np.load(data_loc + '/Data/Cam_Data/rba_data_608x768.npy')
# data_calib =  np.load(data_loc + '/Data/Cam_Data/Cleaned_Data/Calibrations/rba_rz_pos_30280_30360.npz')

res = configuration['Resolution']
# gridx = data_calib['r_pos'][::res, ::res]
# gridy = data_calib['z_pos'][::res, ::res]
u_sol = data.astype(np.float32)[:,:,::res, ::res]

u_2_sol = data_2.astype(np.float32)[:,:,::res,::res]
u_sol = np.vstack((u_sol, u_2_sol))

np.random.shuffle(u_sol)
u_sol = u_sol[:11]
del u_2_sol
# %%

grid_size_x = u_sol.shape[2]
grid_size_y = u_sol.shape[3]

u = torch.from_numpy(u_sol)
u = u.permute(0, 2, 3, 1)


# ntrain = 75
# ntest = 11

ntrain = 25
ntest = 5


S_x = grid_size_x #Grid Size
S_y = grid_size_y #Grid Size

#Extracting hyperparameters from the config dict

modes = configuration['Modes']
width = configuration['Width']
output_size = configuration['Step']
# hidden_size = configuration['Hidden Size']
# num_cells = configuration['Cells']

t1 = default_timer()

T_in = input_size = configuration['T_in']
T = configuration['T_out']
T_out = T
step = output_size = configuration['Step']

modes = configuration['Modes']
width = configuration['Width']

batch_size = configuration['Batch Size']

# %%

################################################################
# Sort Data into test/train sets -- Sequential - Ignorant of shots but only sequences
################################################################

t_sets = T_in + T_out - input_size - output_size

u1 = []
u2 = []
for ii in tqdm(range(len(u))):
    for jj in range(t_sets):
        u1.append(u[ii, :, :, jj:jj+input_size])
        u2.append(u[ii, :, :, jj+input_size:jj+input_size+step])
    
# u1 = np.asarray(u1)
# u2 = np.asarray(u2)

u1 = torch.stack(u1)
u2 = torch.stack(u2)

ntrain = int(75*t_sets)
ntest = len(u1) - ntrain

t1 = default_timer()

test_a = u1
test_u = u2

print('Test Input: ', test_a.shape)
print('Test Output ', test_u.shape)

# %%
#Normalising the train and test datasets with the preferred normalisation. 

norm_strategy = configuration['Normalisation Strategy']

# a_normalizer = MinMax_Normalizer(test_a)
# y_normalizer = MinMax_Normalizer(test_u)

a_normalizer = RangeNormalizer(test_a)
y_normalizer = RangeNormalizer(test_u)

test_a = a_normalizer.encode(test_a)
test_u_norm = y_normalizer.encode(test_u)

# %%

#Using arbitrary R and Z positions sampled uniformly within a specified domain range. 
x_grid = np.linspace(-1.0, -2.0, 400)[::res]
# x = np.linspace(-1.0, -2.0, 608)[::res]

y_grid = np.linspace(0.0, 1.0, 512)[::res]
# y = np.linspace(-1.0, 0.0, 768)[::res]

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u_norm), batch_size=20, shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %%

################################################################
# fourier layer
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)


    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

# class FRNN_Cell(nn.Module):
#    def __init__(self, modes, width, batch_first=True):
#         super(FRNN_Cell, self).__init__()
        
#         self.modes = modes
#         self.width = width
        
#         self.F_x = SpectralConv2d(self.width, self.width, self.modes, self.modes)
#         # self.F_h = SpectralConv2d(self.width, self.width, self.modes, self.modes)

#         self.W_x = nn.Conv2d(self.width, self.width, 1)
#         self.W_h = nn.Conv2d(self.width, self.width, 1)

        
#    def forward(self, x, h):
#         batchsize = x.shape[0]
#         size_x, size_y = x.shape[1], x.shape[2]

#         # h1 = self.F_h(h)
#         h2 = self.W_h(h)
#         h = h2
#         h = F.gelu(h)
        
#         x1 = self.F_x(x)
#         x2 = self.W_x(x)
#         x = x1 + x2
#         x = F.gelu(x)
        
#         h = h+x
#         y = torch.tanh(h)
        
#         return y, h.clone().detach()
    
#    def count_params(self):
#         c = 0
#         for p in self.parameters():
#             c += reduce(operator.mul, list(p.size()))
    
#         return c


# class FRNN(nn.Module):
#    def __init__(self, modes, width, n_output, n_hidden, n_cells, T_in, batch_first=True):
#         super(FRNN, self).__init__()
        
#         self.modes = modes
#         self.width = width
#         self.n_output = n_output 
#         self.n_hidden = n_hidden
        
#         self.linear_in_x = nn.Linear(T_in+2, self.width)
#         self.linear_in_h = nn.Linear(self.n_hidden, self.width)
#         self.linear_out_x = nn.Linear(self.width, self.n_output)
#         self.linear_out_h = nn.Linear(self.width , self.n_hidden-2)

#         self.FRNN_Cells = nn.ModuleList()
        
#         for ii in range(n_cells):
#             self.FRNN_Cells.append(FRNN_Cell(self.modes, self.width))

        
#    def forward(self, x, h):
#         grid = self.get_grid(x.shape, x.device)
#         x = torch.cat((x, grid), dim=-1)
#         h = torch.cat((h, grid), dim=-1)


#         h = self.linear_in_h(h)
#         x = self.linear_in_x(x)

#         h = h.permute(0, 3, 1, 2)
#         x = x.permute(0, 3, 1, 2)

#         for cell in self.FRNN_Cells:
#             x, h = cell(x, h)   

#         h = h.permute(0, 2, 3, 1)
#         x = x.permute(0, 2, 3, 1)

#         y = self.linear_out_x(x)
#         h = self.linear_out_h(h)
#         return y, h.clone().detach()

# #Using x and y values from the simulation discretisation 
#    def get_grid(self, shape, device):
#         batchsize, size_x, size_y = shape[0], shape[1], shape[2]
#         gridx = torch.tensor(x_grid, dtype=torch.float)
#         gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#         gridy = torch.tensor(y_grid, dtype=torch.float)
#         gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
#         return torch.cat((gridx, gridy), dim=-1).to(device)

# ## Arbitrary grid discretisation 
#     # def get_grid(self, shape, device):
#     #     batchsize, size_x, size_y = shape[0], shape[1], shape[2]
#     #     gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#     #     gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#     #     gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
#     #     gridy = gridy.reshape(1, 1, size_y, 1).repeat([batch
  

#    def count_params(self):
#         c = 0
#         for p in self.parameters():
#             c += reduce(operator.mul, list(p.size()))
    
#         return c
        
class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous T_in timesteps + 2 locations (u(t-T_in, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=x_discretistion, y=y_discretisation, c=T_in)
        output: the solution of the next timestep
        output shape: (batchsize, x=x_discretisation, y=y_discretisatiob, c=step)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(T_in+2, self.width)
        # input channel is 12: the solution of the previous T_in timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv5 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        # self.mlp0 = MLP(self.width, self.width, self.width)
        # self.mlp1 = MLP(self.width, self.width, self.width)
        # self.mlp2 = MLP(self.width, self.width, self.width)
        # self.mlp3 = MLP(self.width, self.width, self.width)
        # self.mlp4 = MLP(self.width, self.width, self.width)
        # self.mlp5 = MLP(self.width, self.width, self.width)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.w4 = nn.Conv2d(self.width, self.width, 1)
        self.w5 = nn.Conv2d(self.width, self.width, 1)

        # self.norm = nn.InstanceNorm2d(self.width)
        self.norm = nn.Identity()

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, step)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.norm(self.conv0(self.norm(x)))
        # x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1+x2
        x = F.gelu(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        # x1 = self.mlp1(x1)    
        x2 = self.w1(x)
        x = x1+x2
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        # x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1+x2
        x = F.gelu(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        # x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1+x2

        x1 = self.norm(self.conv4(self.norm(x)))
        # x1 = self.mlp4(x1)
        x2 = self.w4(x)
        x = x1+x2

        x1 = self.norm(self.conv5(self.norm(x)))
        # x1 = self.mlp5(x1)
        x2 = self.w5(x)
        x = x1+x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

#Using x and y values from the simulation discretisation 
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(x_grid, dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(y_grid, dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

## Arbitrary grid discretisation 
    # def get_grid(self, shape, device):
    #     batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    #     gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    #     gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    #     gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    #     gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    #     return torch.cat((gridx, gridy), dim=-1).to(device)

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

# %% 
################################################################
# training and evaluation
################################################################
#FRNN
# frnn = FRNN(modes, width, output_size, hidden_size, num_cells, T_in).to(device)
# frnn.load_state_dict(torch.load(file_loc + '/Models/FRNN_rba_boolean-switch.pth', map_location=torch.device('cpu')))


# FNO
fno  = FNO2d(modes, modes, width)
# fno.load_state_dict(torch.load(file_loc + '/Models/FNO_rba_chamfered-surface.pth', map_location=torch.device('cpu')))
fno.load_state_dict(torch.load(file_loc + '/Models/FNO_rba_dyadic-extra.pth', map_location=torch.device('cpu')))

# run.update_metadata({'Number of Params': int(model.count_params())})
# print("Number of model params : " + str(model.count_params()))

# model = nn.DataParallel(model, device_ids = [0,1])
# model.to(device)

# # wandb.watch(model, log='all')

# optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
myloss = nn.MSELoss()

       
# %%
# #Testing - FRNN
# batch_size = 1 
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)

# pred_set = torch.zeros(test_u.shape)
# index = 0

# with torch.no_grad():
#     for xx, yy in tqdm(test_loader):
#         loss = 0
#         xx = xx.to(device)
#         yy = yy.to(device)
#         hidden = torch.zeros(xx.shape[0],grid_size_x, grid_size_y, hidden_size-2).to(device)

#         for tt in range(t_sets):
#             x = xx[:,tt]
#             y = yy[:,tt]
#             pred, hidden = frnn(x, hidden)     
#             pred_set[index, tt]=pred   
#             loss += myloss(pred.reshape(batch_size, -1), y.reshape(batch_size, -1))
    
# test_l2 = (pred_set - test_u_norm).pow(2).mean()
# print('Testing Error: %.3e' % (test_l2))



# run.update_metadata({'Training Time': float(train_time),
#                      'MSE Test Error': float(test_l2)
#                     })

# pred_set_frnn = y_normalizer.decode(pred_set.to(device)).cpu()

# %%
#Testing - fno

batch_size = 1 
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u_norm), batch_size=1, shuffle=False)

pred_set = torch.zeros(test_u.shape)
index = 0
with torch.no_grad():
    for xx, yy in tqdm(test_loader):
        
        xx = xx.to(device)
        yy = yy.to(device)
        
        pred = fno(xx)
        pred_set[index]=pred
        index += 1
    
test_l2 = (pred_set - test_u_norm).pow(2).mean()
print('Testing Error: %.3e' % (test_l2))


pred_set_fno = y_normalizer.decode(pred_set.to(device)).cpu()
# %%

# idx = np.random.randint(0,ntest) 
idx = 3

u_field = test_u[idx]

v_min_1 = torch.min(u_field[0, :,:])
v_max_1 = torch.max(u_field[0, :,:])

v_min_2 = torch.min(u_field[int(t_sets/2), :, :])
v_max_2 = torch.max(u_field[int(t_sets/2), :, :])

v_min_3 = torch.min(u_field[-1, :, :])
v_max_3 = torch.max(u_field[-1, :, :])

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(2,3,1)
pcm =ax.imshow(u_field[0,:,:], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
# ax.title.set_text('Initial')
ax.title.set_text('t='+ str(T_in+step))
ax.set_ylabel('Solution')
fig.colorbar(pcm, pad=0.05)


ax = fig.add_subplot(2,3,2)
pcm = ax.imshow(u_field[int(t_sets/2),:,:], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2, vmax=v_max_2)
# ax.title.set_text('Middle')
ax.title.set_text('t='+ str(int((T_in+t_sets/2 +step))))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)


ax = fig.add_subplot(2,3,3)
pcm = ax.imshow(u_field[-1,:,:], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
# ax.title.set_text('Final')
ax.title.set_text('t='+str(T_in + step + t_sets))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)


# u_field = pred_set_frnn[idx]
# # u_field = pred_set_fno[idx]

# ax = fig.add_subplot(2,3,4)
# pcm = ax.imshow(u_field[0,:,:,-1], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
# ax.set_ylabel('FRNN')

# fig.colorbar(pcm, pad=0.05)

# ax = fig.add_subplot(2,3,5)
# pcm = ax.imshow(u_field[int(t_sets/2),:,:,-1], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2, vmax=v_max_2)
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# fig.colorbar(pcm, pad=0.05)


# ax = fig.add_subplot(2,3,6)
# pcm = ax.imshow(u_field[-1,:,:,-1], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# fig.colorbar(pcm, pad=0.05)

# output_plot = file_loc + '/Plots/FRNN_rba_' + run.name + '.png'
# plt.savefig(output_plot)

# %% 

# CODE = ['FRNN_rba.py']
# INPUTS = []
# OUTPUTS = [model_loc, output_plot]

# # Save code files
# for code_file in CODE:
#     if os.path.isfile(code_file):
#         run.save(code_file, 'code')
#     elif os.path.isdir(code_file):
#         run.save_directory(code_file, 'code', 'text/plain', preserve_path=True)
#     else:
#         print('ERROR: code file %s does not exist' % code_file)


# # Save input files
# for input_file in INPUTS:
#     if os.path.isfile(input_file):
#         run.save(input_file, 'input')
#     elif os.path.isdir(input_file):
#         run.save_directory(input_file, 'input', 'text/plain', preserve_path=True)
#     else:
#         print('ERROR: input file %s does not exist' % input_file)


# # Save output files
# for output_file in OUTPUTS:
#     if os.path.isfile(output_file):
#         run.save(output_file, 'output')
#     elif os.path.isdir(output_file):
#         run.save_directory(output_file, 'output', 'text/plain', preserve_path=True)   
#     else:
#         print('ERROR: output file %s does not exist' % output_file)

# run.close()
