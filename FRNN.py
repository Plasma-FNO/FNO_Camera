#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 6 Jan 2023
@author: vgopakum
FNO modelled over the MHD data built using JOREK for multi-blob diffusion. 
"""
# %%
configuration = {"Case": 'Gyrokinetics', #Specifying the Simulation Scenario
                 "Field": 'T', #Variable we are modelling
                 "Type": 'FRNN', #FNO Architecture
                 "Epochs": 500, 
                 "Batch Size": 20,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.001,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GELU',
                 "Normalisation Strategy": 'Min-Max',
                 "Instance Norm": 'No', #Layerwise Normalisation
                 "Log Normalisation":  'No',
                 "Physics Normalisation": 'Yes', #Normalising the Variable 
                 "T_in": 10, #Input time steps
                 "T_out": 40, #Max simulation time
                 "Step": 5, #Time steps output in each forward call
                 "Modes": 8, #Number of Fourier Modes
                 "Width": 16, #Features of the Convolutional Kernel
                 "Hidden Size":16,
                 "Cells": 1,
                 "Variables": 1, 
                 "Noise": 0.0, 
                 "Loss Function": 'LP Loss' #Choice of Loss Function
                 }

## %% 
#Simvue Setup. If not using comment out this section and anything with run
from simvue import Run
run = Run()
run.init(folder="/GyroKinetics", tags=['FRNN', 'GyroKinetics', 'GX', 'ExB', 'Tperp'], metadata=configuration)


# %%
#Importing the necessary packages. 
import os 
import sys
import pdb

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
from tqdm import tqdm 

torch.manual_seed(0)
np.random.seed(0)

# %% 
#Setting up the directories - data location, model location and plots. 
path = os.getcwd()
data_loc = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
# model_loc = os.path.dirname(os.path.dirname(os.getcwd()))
file_loc = os.getcwd()


# %%
#Setting up CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

restart = int(sys.argv[1])

# %%
##################################
#Normalisation Functions 
##################################


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

# normalization, Gaussian - across the entire dataset
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


# normalization, scaling by range - pointwise
class RangeNormalizer(object):
    def __init__(self, x, low=-1.0, high=1.0):
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

#normalization, rangewise but across the full domain 
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


# %%
##################################
# Loss Functions
##################################

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

# #Adding Gaussian Noise to the training dataset
# class AddGaussianNoise(object):
#     def __init__(self, mean=0., std=1.):
#         self.mean = torch.FloatTensor([mean])
#         self.std = torch.FloatTensor([std])
        
#     def __call__(self, tensor):
#         return tensor + torch.randn(tensor.size()).cuda() * self.std + self.mean
    
#     def __repr__(self):
#         return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
#     def cuda(self):
#         self.mean = self.mean.cuda()
#         self.std = self.std.cuda()
#     def cpu(self):
#         self.mean = self.mean.cpu()
#         self.std = self.std.cpu()
# # additive_noise = AddGaussianNoise(0.0, configuration['Noise'])
# additive_noise.cuda()

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

class FRNN_Cell(nn.Module):
   def __init__(self, modes, width, batch_first=True):
        super(FRNN_Cell, self).__init__()
        
        self.modes = modes
        self.width = width
        
        self.F_x = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        # self.F_h = SpectralConv2d(self.width, self.width, self.modes, self.modes)

        self.W_x = nn.Conv2d(self.width, self.width, 1)
        self.W_h = nn.Conv2d(self.width, self.width, 1)

        
   def forward(self, x, h):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        # h1 = self.F_h(h)
        h2 = self.W_h(h)
        h = h2
        h = F.gelu(h)
        
        x1 = self.F_x(x)
        x2 = self.W_x(x)
        x = x1 + x2
        x = F.gelu(x)
        
        h = h+x
        y = torch.tanh(h)
        
        return y, h.clone().detach()
    
   def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
    
        return c


class FRNN(nn.Module):
   def __init__(self, modes, width, n_output, n_hidden, n_cells, T_in, batch_first=True):
        super(FRNN, self).__init__()
        
        self.modes = modes
        self.width = width
        self.n_output = n_output 
        self.n_hidden = n_hidden
        
        self.linear_in_x = nn.Linear(T_in+2, self.width)
        self.linear_in_h = nn.Linear(self.n_hidden, self.width)
        self.linear_out_x = nn.Linear(self.width, self.n_output)
        self.linear_out_h = nn.Linear(self.width , self.n_hidden)

        self.FRNN_Cells = nn.ModuleList()
        
        for ii in range(n_cells):
            self.FRNN_Cells.append(FRNN_Cell(self.modes, self.width))

        
   def forward(self, x, h):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        # h = torch.cat((h, grid), dim=-1)


        h = self.linear_in_h(h)
        x = self.linear_in_x(x)

        h = h.permute(0, 3, 1, 2)
        x = x.permute(0, 3, 1, 2)

        for cell in self.FRNN_Cells:
            x, h = cell(x, h)   

        h = h.permute(0, 2, 3, 1)
        x = x.permute(0, 2, 3, 1)

        y = self.linear_out_x(x)
        h = self.linear_out_h(h)
        return y, h.clone().detach()

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
    #     gridy = gridy.reshape(1, 1, size_y, 1).repeat([batch
  

   def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
    
        return c

# %%

################################################################
# Loading Data 
################################################################



#x_grid, y_grid = np.meshgrid(np.linspace(-0., 1.0, 64),np.linspace(-0., 1.0, 64))
x_grid = np.linspace(-0., 1.0, 64)
y_grid = np.linspace(-0., 1.0, 64)


# u_sol  = np.load(data_loc + "/Data/vEx.npy")
u_sol  = np.load(data_loc + "/Data/Tperp.npy")



ntrain = 150
ntest = 40
S = 64 #Grid Size 


#Extracting hyperparameters from the config dict
modes = configuration['Modes']
width = configuration['Width']
output_size = configuration['Step']
hidden_size = configuration['Hidden Size']
num_cells = configuration['Cells']
batch_size = configuration['Batch Size']
T_in = configuration['T_in']
T = configuration['T_out']
step = configuration['Step']

t1 = default_timer()

np.random.shuffle(u_sol)
u = torch.from_numpy(u_sol).float()
#u = u.permute(0, 2, 3, 1) # RG: changed it here due to a difference in order
u = u.permute(3, 1, 2, 0)

#At this stage the data needs to be [Batch_Size, X, Y, T]
# Training input
train_a = u[:ntrain,:,:,:T_in]
# Training output
train_u = u[:ntrain,:,:,T_in:T+T_in]

test_a = u[-ntest:,:,:,:T_in]
test_u = u[-ntest:,:,:,T_in:T+T_in]

group_size = int(20)
test_ul = torch.zeros((2, 64, 64, 800))

for i in range(int(int(T)/group_size)):
    for j in range(group_size):
        test_ul[i, :, :, j*T:(j+1)*T] = test_u[i*group_size+j, :, :, :]

#test_al = ul[-1:,:,:,:T_in]
#test_ul = ul[-1:,:,:,T_in:Tl+T_in]

print(train_u.shape)
print(test_u.shape)
#print(test_ul.shape)


# %%
#Normalising the train and test datasets with the preferred normalisation. 

norm_strategy = configuration['Normalisation Strategy']

if norm_strategy == 'Min-Max':
    a_normalizer = MinMax_Normalizer(train_a)
    y_normalizer = MinMax_Normalizer(train_u)

if norm_strategy == 'Range':
    a_normalizer = RangeNormalizer(train_a)
    y_normalizer = RangeNormalizer(train_u)

if norm_strategy == 'Gaussian':
    a_normalizer = GaussianNormalizer(train_a)
    y_normalizer = GaussianNormalizer(train_u)



train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

train_u = y_normalizer.encode(train_u)
test_u_encoded = y_normalizer.encode(test_u)

test_al = test_a[:2]
test_ul_encoded = torch.zeros((2, 64, 64, 800))


for i in range(int(int(T)/group_size)):
    for j in range(group_size):
        test_ul_encoded[i, :, :, j*T:(j+1)*T] = test_u_encoded[i*group_size+j, :, :, :]

#pdb.set_trace()
#print(test_u_encoded[4, :, :, :] - test_ul_encoded[1, :, :, 40:80])

#pdb.set_trace()
#test_al = a_normalizer.encode(test_al)
#test_ul_encoded = y_normalizer.encode(test_ul)

# %%
#Setting up the dataloaders for the test and train datasets. 
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u_encoded), batch_size=batch_size, shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)

# %%

################################################################
# training and evaluation
################################################################

#Instantiating the Model. 
model = FRNN(modes, width, hidden_size, num_cells, T_in, output_size, x_grid, y_grid).to(device)
model.to(device)

run.update_metadata({'Number of Params': int(model.count_params())})
print("Number of model params : " + str(model.count_params()))

#Setting up the optimisation schedule. 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])

myloss = LpLoss(size_average=False)

epochs = configuration['Epochs']
if torch.cuda.is_available():
    y_normalizer.cuda()

# %%

#Training Loop
start_time = time.time()
#for ep in tqdm(range(epochs)): #Training Loop - Epochwise
for ep in range(epochs): #Training Loop - Epochwise
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader: #Training Loop - Batchwise
        optimizer.zero_grad()
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        # xx = additive_noise(xx)
        hidden = torch.zeros(xx.shape[0], grid_size_x, grid_size_y, hidden_size).to(device).detach()

        for t in range(0, T, step): #Training Loop - Time rollouts. 
            y = yy[..., t:t + step]
            im, hidden = model(xx, hidden)       

            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1)) 

            #Storing the rolled out outputs. 
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            #Preparing the autoregressive input for the next time step. 
            xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        loss.backward()
        # train_l2_full.backward()
        optimizer.step()

#Validation Loop
    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            hidden = torch.zeros(xx.shape[0],grid_size_x, grid_size_y, hidden_size).to(device).detach()

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im, hidden = model(xx, hidden)       
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    t2 = default_timer()
    scheduler.step()

    train_loss = train_l2_full / ntrain
    test_loss = test_l2_full / ntest

    print('Epochs: %d, Time: %.2f, Train Loss per step: %.3e, Train Loss: %.3e, Test Loss per step: %.3e, Test Loss: %.3e' % (ep, t2 - t1, train_l2_step / ntrain / (T / step), train_loss, test_l2_step / ntest / (T / step), test_loss))

    run.log_metrics({'Train Loss': train_loss, 
                   'Test Loss': test_loss})

train_time = time.time() - start_time
# %%
#Saving the Model
# model_loc = file_loc + '/Models/GX_' + "vEx" + '.pth'
model_loc = file_loc + '/Models/GX_' + "TPerp" + '.pth'
torch.save(model.state_dict(),  model_loc)

# %%
#Testing 
batch_size = 1
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_al, test_ul_encoded), batch_size=1, shuffle=False)
pred_set = torch.zeros(test_u.shape)
index = 0
with torch.no_grad():
    #for xx, yy in tqdm(test_loader):
    for xx, yy in test_loader:
        loss = 0
        xx, yy = xx.to(device), yy.to(device)
        # xx = additive_noise(xx)
        hidden = torch.zeros(xx.shape[0],grid_size_x, grid_size_y, hidden_size).to(device).detach()
        t1 = default_timer()
        for t in range(0, T, step):
            y = yy[..., t:t + step]
            out, hidden = model(xx, hidden)       
            loss += myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1))
            # loss += myloss(out.reshape(batch_size, -1)*torch.log(out.reshape(batch_size, -1)), y.reshape(batch_size, -1)*torch.log(y.reshape(batch_size, -1)))

            if t == 0:
                pred = out
            else:
                pred = torch.cat((pred, out), -1)       

            xx = torch.cat((xx[..., step:], out), dim=-1)

        t2 = default_timer()
        # pred = y_normalizer.decode(pred)
        pred_set[index]=pred
        index += 1
        print(t2-t1, loss)

# %%
#Logging Metrics 
MSE_error = (pred_set - test_ul_encoded).pow(2).mean()
MAE_error = torch.abs(pred_set - test_ul_encoded).mean()
LP_error = loss / (ntest*T/step)

print('(MSE) Testing Error: %.3e' % (MSE_error))
print('(MAE) Testing Error: %.3e' % (MAE_error))
print('(LP) Testing Error: %.3e' % (LP_error))

run.update_metadata({'Training Time': float(train_time),
                    'MSE Test Error': float(MSE_error),
                    'MAE Test Error': float(MAE_error),
                    'LP Test Error': float(LP_error)
                   })

pred_set = y_normalizer.decode(pred_set.to(device)).cpu()

# %%
#Plotting the comparison plots

idx = np.random.randint(0,ntest) 
idx = 0

if configuration['Log Normalisation'] == 'Yes':
    test_u = torch.exp(test_u)
    pred_set = torch.exp(pred_set)

u_field = test_ul[idx]

v_min_1 = torch.min(u_field[:,:,0])
v_max_1 = torch.max(u_field[:,:,0])

v_min_2 = torch.min(u_field[:, :, int(group_size*T/2)])
v_max_2 = torch.max(u_field[:, :, int(group_size*T/2)])

v_min_3 = torch.min(u_field[:, :, -1])
v_max_3 = torch.max(u_field[:, :, -1])

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(2,3,1)
pcm =ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
# ax.title.set_text('Initial')
ax.title.set_text('t='+ str(T_in))
ax.set_ylabel('Solution')
fig.colorbar(pcm, pad=0.05)


ax = fig.add_subplot(2,3,2)
pcm = ax.imshow(u_field[:,:,int(group_size*T/2)], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2, vmax=v_max_2)
# ax.title.set_text('Middle')
ax.title.set_text('t='+ str(int((group_size*T+T_in)/2)))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)


ax = fig.add_subplot(2,3,3)
pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
# ax.title.set_text('Final')
ax.title.set_text('t='+str(group_size*T+T_in))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)


u_field = pred_set[idx]

ax = fig.add_subplot(2,3,4)
pcm = ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
ax.set_ylabel('FNO')

fig.colorbar(pcm, pad=0.05)

ax = fig.add_subplot(2,3,5)
pcm = ax.imshow(u_field[:,:,int(group_size*T/2)], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2, vmax=v_max_2)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)


ax = fig.add_subplot(2,3,6)
pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
fig.colorbar(pcm, pad=0.05)


# %%
# output_plot = file_loc + '/vEx.png'
output_plot = file_loc + '/Tperp.png'

plt.savefig(output_plot)

# %%
#Simvue Artifact storage
CODE = ['FRNN.py']
INPUTS = []
OUTPUTS = [model_loc, output_plot]

# Save code files
for code_file in CODE:
   if os.path.isfile(code_file):
       run.save(code_file, 'code')
   elif os.path.isdir(code_file):
       run.save_directory(code_file, 'code', 'text/plain', preserve_path=True)
   else:
       print('ERROR: code file %s does not exist' % code_file)


# Save input files
for input_file in INPUTS:
   if os.path.isfile(input_file):
       run.save(input_file, 'input')
   elif os.path.isdir(input_file):
       run.save_directory(input_file, 'input', 'text/plain', preserve_path=True)
   else:
       print('ERROR: input file %s does not exist' % input_file)


# Save output files
for output_file in OUTPUTS:
   if os.path.isfile(output_file):
       run.save(output_file, 'output')
   elif os.path.isdir(output_file):
       run.save_directory(output_file, 'output', 'text/plain', preserve_path=True)   
   else:
       print('ERROR: output file %s does not exist' % output_file)

run.close()
