#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 6 Jan 2023
@author: vgopakum
FNO modelled over Camera data = rbb camera looking at the central solenoid
"""
# %%
configuration = {"Case": 'RBB Camera', #Specifying the Camera setup
                 "Pipeline": 'Sequential', #Shot-Agnostic RNN windowed data pipeline. 
                 "Calibration": 'Invariant', #CAD inspired Geometry setup
                 "Epochs": 250, 
                 "Batch Size": 20,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GELU',
                 "Normalisation Strategy": 'Min-Max',
                 "Instance Norm": 'No', #Layerwise Normalisation
                 "Log Normalisation":  'No',
                 "Physics Normalisation": 'No', #Normalising the Variable 
                 "T_in": 10, #Input time steps
                 "T_out": 'All', #Max simulation time
                 "Step": 10, #Time steps output in each forward call
                 "Modes":8, #Number of Fourier Modes
                 "Width": 16, #Features of the Convolutional Kernel
                 "Loss Function": 'LP-Loss', #Choice of Loss Fucnction
                 "Resolution":1
                 }

## %% 
#Simvue Setup. If not using comment out this section and anything with run
from simvue import Run
run = Run(mode='disabled')
run.init(folder="/FNO_Camera", tags=['FNO', 'Camera', 'rbb', 'Forecasting', 'shot-agnostic', 'discretisation-invariant', 'Full Length'], metadata=configuration)

# %%
#Importing the necessary packages. 
import numpy as np
from tqdm import tqdm 
import h5py
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
import os 
path = os.getcwd()
# data_loc = '/home/ir-gopa2/rds/rds-ukaea-ap001/ir-gopa2/Data/Cam_Data/rbb_30255_30431' #CSD3
data_loc = '/Users/Vicky/Documents/UKAEA/Data/Camera_Data/rbb_30255_30431'
model_loc = '/Users/Vicky/Documents/UKAEA/Code/FNO/Camera_Forecasting_Plots/Models'
file_loc = os.getcwd()

# %%
#Setting up CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
##################################
#Normalisation Functions
##################################

#normalization, rangewise but across the full domain by the max and minimum of the camera image. 
class MinMax_Normalizer(object):
    def __init__(self, low=-1.0, high=1.0):
        super(MinMax_Normalizer, self).__init__()
        # self.mymin = torch.min(x)
        # self.mymax = torch.max(x)
        self.mymin = torch.tensor(0.0)
        self.mymax =torch.tensor(255.0)

        self.a = (high - low)/(self.mymax - self.mymin)
        self.b = -self.a*self.mymax + high

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
# Loss Functionss
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
################################################################
# fourier layer - Setting up the spectral convoltions
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
        x2 = self.w0(x)
        x = x1+x2
        x = F.gelu(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x2 = self.w1(x)
        x = x1+x2
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x2 = self.w2(x)
        x = x1+x2
        x = F.gelu(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x2 = self.w3(x)
        x = x1+x2

        x1 = self.norm(self.conv4(self.norm(x)))
        x2 = self.w4(x)
        x = x1+x2

        x1 = self.norm(self.conv5(self.norm(x)))
        x2 = self.w5(x)
        x = x1+x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

#Using x and y values discretised along the view of the camera. 
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(-1.5, 1.5, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(-2.0, 2.0, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

# # Arbitrary grid discretisation 
#     def get_grid(self, shape, device):
#         batchsize, size_x, size_y = shape[0], shape[1], shape[2]
#         gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#         gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#         gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
#         gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
#         return torch.cat((gridx, gridy), dim=-1).to(device)

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

# %%

################################################################
# Loading Data 
################################################################

#  30255 - 30431 : Initial RBB Camera Data

# %% 
#Function that takes in shot numbers, retreives the camera data within that, 
#chunks it into time windows, converts to a tensor and permutes to FNO friendly shapes. 

def time_windowing(shots):
    u1 = [] #input a
    u2 = [] #output u
    for ii in shots:
        data = h5py.File(data_loc + '/'+'rbb'+str(ii)+'.h5', 'r')
        data_length = int((len(data.keys()) - 3)/2)
        temp_cam_data = []
        for jj in range(data_length):
            if jj<10:
                temp_cam_data.append(np.asarray(data['frame000'+str(jj)]))
            if jj>=10 and jj < 100:
                temp_cam_data.append(np.asarray(data['frame00' + str(jj)]))
            if jj >= 100:
                temp_cam_data.append(np.asarray(data['frame0' + str(jj)]))
        
        temp_cam_data = np.asarray(temp_cam_data[5:]) #Removes the first 5 frames. 
        for ff in tqdm(range(len(temp_cam_data) - T_in - step)):
            u1.append(temp_cam_data[ff:ff+input_size, :, :])
            u2.append(temp_cam_data[ff+input_size:ff+input_size+step, :, :])

    u1 = torch.tensor(np.asarray(u1)).permute(0, 2, 3, 1)
    u2 = torch.tensor(np.asarray(u2)).permute(0, 2, 3, 1)
    del temp_cam_data, data
    return u1, u2
    

T_in = input_size = configuration['T_in']
T = T_out = configuration['T_out']
step = output_size = configuration['Step']

shots = np.load(data_loc + '/shotnums_rbb_30255_30428.npy')
# shots = np.sort(shots)
# %%
#Extracting hyperparameters from the config dict
modes = configuration['Modes']
width = configuration['Width']
output_size = configuration['Step']
batch_size = configuration['Batch Size']
res = configuration['Resolution']

t1 = default_timer()


# %%
#Normalising the train and test datasets with the preferred normalisation. 

norm_strategy = configuration['Normalisation Strategy']

if norm_strategy == 'Min-Max':
    normalizer = MinMax_Normalizer()

# if norm_strategy == 'Range':
#     a_normalizer = RangeNormalizer(train_a)
#     y_normalizer = RangeNormalizer(train_u)

# if norm_strategy == 'Gaussian':
#     a_normalizer = GaussianNormalizer(train_a)
#     y_normalizer = GaussianNormalizer(train_u)


# %%
#50 shots are selected for testing over which we perform time windowing and terate over in 5 groups of 10. 
#Preparing the Testing data - the last 5 shots from the curated list
#Training data is prepared mid training. 

test_shots = shots[-5:]
test_a, test_u = time_windowing(test_shots)
test_a = normalizer.encode(test_a)
test_u_encoded = normalizer.encode(test_u)
ntest = len(test_a)

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u_encoded), batch_size=batch_size, shuffle=True)

# %%
#Instantiating the Model. 
model = FNO2d(modes, modes, width)
# model = model.double()
# model = nn.DataParallel(model, device_ids = [0,1])
model.to(device)

run.update_metadata({'Number of Params': int(model.count_params())})
print("Number of model params : " + str(model.count_params()))


#Setting up the optimisation schedule. 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])


#Loading from Checkpoint 

# #Loading from Checkpoint 
# model_name = 'fundamental-vocoder' # 250 8x16
# model_name = 'seething-echelon' # 250 16x32
model_name = 'hard-frame' # 500 8 x 16
# model_name = 'searing-tube' # 500 16 x 32 

if device == torch.device('cpu'):
    checkpoint = torch.load(model_loc + '/FNO_rbb_' + model_name + '.pth', map_location=torch.device('cpu')) #First 250. 
else:
    checkpoint = torch.load(model_loc + '/FNO_rbb_' + model_name + '.pth') #First 250. 

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])


myloss = LpLoss(size_average=False)
# myloss = nn.MSELoss()

epochs = configuration['Epochs']


# %%
#Testing 
batch_size = 1 
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)

pred_set = torch.zeros(test_u.shape)
index = 0

with torch.no_grad():
    for xx, yy in tqdm(test_loader):
                
        xx = xx.to(device)
        yy = yy.to(device)

        pred = model(xx)
        pred_set[index]=pred
        index += 1
    
test_l2 = (pred_set - test_u_encoded).pow(2).mean()
print('Testing Error: %.3e' % (test_l2))
    
# %%

#De-normalising the values
pred_set = normalizer.decode(pred_set.to(device)).cpu()

# %%
import matplotlib as mpl

# plt.rcParams['grid.linewidth'] = 0.5
# plt.rcParams['grid.alpha'] = 0.5
# plt.rcParams['grid.linestyle'] = '-'
# mpl.rcParams['xtick.minor.visible']=True
# mpl.rcParams['font.size']=20
mpl.rcParams['figure.figsize']=(16,12)
# mpl.rcParams['xtick.minor.visible']=True
# mpl.rcParams['axes.linewidth']= 1
# mpl.rcParams['axes.titlepad'] = 30
# plt.rcParams['xtick.major.size'] = 20
# plt.rcParams['ytick.major.size'] = 20
# plt.rcParams['xtick.minor.size'] = 10.0
# plt.rcParams['ytick.minor.size'] = 10.0
# plt.rcParams['xtick.major.width'] = 0.8
# plt.rcParams['ytick.major.width'] = 0.8
# plt.rcParams['xtick.minor.width'] = 0.6
# plt.rcParams['ytick.minor.width'] = 0.6
# mpl.rcParams['lines.linewidth'] = 1
# plt.figure()
# %%
#Plotting the comparison plots -- Selecting a single shot and then a random 10 time instance length from the time window'd portion of that shot. 
#Plotting the predictions within one forward pass : 1 - 10 frames. 
from mpl_toolkits.axes_grid1 import make_axes_locatable

test_shot_lens = [224, 171, 198, 145, 140]
shot_idx = np.random.randint(0,5) #Selecting a random shot from the test range. 
shot_idx = 4
time_idx =  np.random.randint(0,test_shot_lens[shot_idx])
time_idx = 67
idx = int(np.sum(test_shot_lens[:shot_idx]) + time_idx )

aspect_ratio = 0.75

u_field = test_u[idx]

v_min_1 = torch.min(u_field[:,:,0])
v_max_1 = torch.max(u_field[:,:,0])

v_min_2 = torch.min(u_field[:, :, int(step/2)])
v_max_2 = torch.max(u_field[:, :, int(step/2)])

v_min_3 = torch.min(u_field[:, :, -1])
v_max_3 = torch.max(u_field[:, :, -1])

plt.style.use('default')
fig = plt.figure()
mpl.rcParams['figure.figsize']=(16,9)
# Calculate the height ratios based on the aspect ratio
gs = mpl.gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[2,2])
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)

ax = fig.add_subplot(2,3,1)
pcm =ax.imshow(u_field[:,:,0], cmap='plasma', extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_1, vmax=v_max_1, aspect=aspect_ratio)
# ax.title.set_text('Initial')
ax.title.set_text('t='+ str(round(1.2*(time_idx+0), 1) + 18)+'ms')
ax.set_ylabel('Solution')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(2,3,2)
pcm = ax.imshow(u_field[:,:,int(step/2)], cmap='plasma', extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_2, vmax=v_max_2, aspect=aspect_ratio)
# ax.title.set_text('Middle')
ax.title.set_text('t='+ str(round(1.2*(time_idx+5), 1) + 18)+'ms')
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(2,3,3)
pcm = ax.imshow(u_field[:,:,-1], cmap='plasma',  extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_3, vmax=v_max_3, aspect=aspect_ratio)
# ax.title.set_text('Final'
ax.title.set_text('t='+ str(round(1.2*(time_idx+10), 1) + 18)+'ms')
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))
u_field = pred_set[idx]

ax = fig.add_subplot(2,3,4)
pcm = ax.imshow(u_field[:,:,0], cmap='plasma', extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_1, vmax=v_max_1, aspect=aspect_ratio)
ax.set_ylabel('FNO')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(2,3,5)
pcm = ax.imshow(u_field[:,:,int(step/2)], cmap='plasma',  extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_2, vmax=v_max_2, aspect=aspect_ratio)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(2,3,6)
pcm = ax.imshow(u_field[:,:,-1], cmap='plasma',  extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_3, vmax=v_max_3, aspect=aspect_ratio)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

plt.tight_layout()

# plt.savefig(file_loc + '/rbb_feedforward10_hard-frame.pdf')


# output_plot = file_loc + '/Plots/rbb_' + run.name + '.png'
# plt.savefig(output_plot)

# plt.title(test_shots[-5:][shot_idx])

# %%
# Plotting the 10th frame across the shot 
from mpl_toolkits.axes_grid1 import make_axes_locatable


u_field = test_u[int(np.sum(test_shot_lens[:shot_idx])):int(np.sum(test_shot_lens[:shot_idx+1])), :, :, :]

v_min_1 = torch.min(u_field[0, :,:,-1])
v_max_1 = torch.max(u_field[0, :,:,-1])

v_min_2 = torch.min(u_field[int(test_shot_lens[shot_idx]/2), :, :, -1])
v_max_2 = torch.max(u_field[int(test_shot_lens[shot_idx]/2), :, :, -1])

v_min_3 = torch.min(u_field[-1, :, :, -1])
v_max_3 = torch.max(u_field[-1, :, :, -1])

plt.style.use('default')
fig = plt.figure()
mpl.rcParams['figure.figsize']=(16,9)
# Calculate the height ratios based on the aspect ratio
gs = mpl.gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[2,2])
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)

ax = fig.add_subplot(2,3,1)
pcm =ax.imshow(u_field[0,:,:,-1], cmap='plasma', extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_1, vmax=v_max_1, aspect=aspect_ratio)
# ax.title.set_text('Initial')
ax.title.set_text('t='+ str((T_in+5+step)*1.2))
ax.set_ylabel('Solution')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(2,3,2)
pcm = ax.imshow(u_field[int(len(u_field)/2),:,:,-1], cmap='plasma', extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_2, vmax=v_max_2, aspect=aspect_ratio)
# ax.title.set_text('Middle')
ax.title.set_text('t='+ str((len(u_field)/2 +  T_in + 5 + step )*1.2))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(2,3,3)
pcm = ax.imshow(u_field[-1, :,:,-1], cmap='plasma',  extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_3, vmax=v_max_3, aspect=aspect_ratio)
# ax.title.set_text('Final')
ax.title.set_text('t='+ str((len(u_field)+  T_in + 5 + step )*1.2))
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

# u_field = pred_set_reshaped[idx]
u_field = pred_set[int(np.sum(test_shot_lens[:shot_idx])):int(np.sum(test_shot_lens[:shot_idx+1])), :, :, :]

ax = fig.add_subplot(2,3,4)
pcm = ax.imshow(u_field[0,:,:,-1], cmap='plasma', extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_1, vmax=v_max_1, aspect=aspect_ratio)
ax.set_ylabel('FNO')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(2,3,5)
pcm = ax.imshow(u_field[int(len(u_field)/2),:,:,-1], cmap='plasma',  extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_2, vmax=v_max_2, aspect=aspect_ratio)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(2,3,6)
pcm = ax.imshow(u_field[-1,:,:,-1], cmap='plasma',  extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_3, vmax=v_max_3, aspect=aspect_ratio)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))


plt.tight_layout()


# plt.savefig(file_loc + '/rbb_last10_hard-frame.pdf')


# %%
# Error Plots
#Plotting the comparison plots -- Selecting a single shot and then a random 10 time instance length from the time window'd portion of that shot. 
#Plotting the predictions within one forward pass : 1 - 10 frames. 
from mpl_toolkits.axes_grid1 import make_axes_locatable

test_shot_lens = [224, 171, 198, 145, 140]
shot_idx = np.random.randint(0,5) #Selecting a random shot from the test range. 
shot_idx = 4
time_idx =  np.random.randint(0,test_shot_lens[shot_idx])
time_idx = 45
idx = int(np.sum(test_shot_lens[:shot_idx]) + time_idx )
mode = 'Lmode'

#Taking the H-mode
shot_idx = 4
time_idx =  np.random.randint(0,test_shot_lens[shot_idx])
time_idx = 121
idx = int(np.sum(test_shot_lens[:shot_idx]) + time_idx )
mode = 'Hmode'


aspect_ratio = 'auto'

u_field = test_u[idx]

v_min_1 = torch.min(u_field[:,:,0])
v_max_1 = torch.max(u_field[:,:,0])

v_min_2 = torch.min(u_field[:, :, int(step/2)])
v_max_2 = torch.max(u_field[:, :, int(step/2)])

v_min_3 = torch.min(u_field[:, :, -1])
v_max_3 = torch.max(u_field[:, :, -1])

plt.style.use('default')
fig = plt.figure()
mpl.rcParams['figure.figsize']=(16,9)
# Calculate the height ratios based on the aspect ratio
gs = mpl.gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[2,2])
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)

ax = fig.add_subplot(3,3,1)
pcm =ax.imshow(u_field[:,:,0], cmap='plasma', extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_1, vmax=v_max_1, aspect=aspect_ratio)
# ax.title.set_text('Initial')
ax.title.set_text('t='+ str(round(1.2*(time_idx+0), 1) + 18)+'ms')
ax.set_ylabel('Solution')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))


ax = fig.add_subplot(3,3,2)
pcm = ax.imshow(u_field[:,:,int(step/2)], cmap='plasma', extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_2, vmax=v_max_2, aspect=aspect_ratio)
# ax.title.set_text('Middle')
ax.title.set_text('t='+ str(round(1.2*(time_idx+5), 1) + 18)+'ms')
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(3,3,3)
pcm = ax.imshow(u_field[:,:,-1], cmap='plasma',  extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_3, vmax=v_max_3, aspect=aspect_ratio)
# ax.title.set_text('Final'
ax.title.set_text('t='+ str(round(1.2*(time_idx+10), 1) + 18)+'ms')
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))


u_field = pred_set[idx]

ax = fig.add_subplot(3,3,4)
pcm = ax.imshow(u_field[:,:,0], cmap='plasma', extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_1, vmax=v_max_1, aspect=aspect_ratio)
ax.set_ylabel('FNO')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(3,3,5)
pcm = ax.imshow(u_field[:,:,int(step/2)], cmap='plasma',  extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_2, vmax=v_max_2, aspect=aspect_ratio)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(3,3,6)
pcm = ax.imshow(u_field[:,:,-1], cmap='plasma',  extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_3, vmax=v_max_3, aspect=aspect_ratio)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

u_field = torch.abs(test_u[idx] - pred_set[idx])

ax = fig.add_subplot(3,3,7)
pcm = ax.imshow(u_field[:,:,0], cmap='plasma', extent=[-1.5, 1.5, -2.0, 2.0], aspect=aspect_ratio)
ax.set_ylabel('Abs Error')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(3,3,8)
pcm = ax.imshow(u_field[:,:,int(step/2)], cmap='plasma',  extent=[-1.5, 1.5, -2.0, 2.0], aspect=aspect_ratio)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(3,3,9)
pcm = ax.imshow(u_field[:,:,-1], cmap='plasma',  extent=[-1.5, 1.5, -2.0, 2.0], aspect=aspect_ratio)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))
plt.tight_layout()

plt.savefig(file_loc + '/rbb_feedforward10_'+model_name+'_error_'+mode+'.pdf')
plt.savefig(file_loc + '/rbb_feedforward10_'+model_name+'_error_'+mode+'.svg')

# %%
#Error Plots

# Plotting the 10th frame across the shot 
from mpl_toolkits.axes_grid1 import make_axes_locatable


u_field = test_u[int(np.sum(test_shot_lens[:shot_idx])):int(np.sum(test_shot_lens[:shot_idx+1])), :, :, :]

v_min_1 = torch.min(u_field[0, :,:,-1])
v_max_1 = torch.max(u_field[0, :,:,-1])

v_min_2 = torch.min(u_field[int(test_shot_lens[shot_idx]/2), :, :, -1])
v_max_2 = torch.max(u_field[int(test_shot_lens[shot_idx]/2), :, :, -1])

v_min_3 = torch.min(u_field[-1, :, :, -1])
v_max_3 = torch.max(u_field[-1, :, :, -1])

plt.style.use('default')
fig = plt.figure()
mpl.rcParams['figure.figsize']=(16,9)
# Calculate the height ratios based on the aspect ratio
gs = mpl.gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[2,2])
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)

ax = fig.add_subplot(3,3,1)
pcm =ax.imshow(u_field[0,:,:,-1], cmap='plasma', extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_1, vmax=v_max_1, aspect=aspect_ratio)
# ax.title.set_text('Initial')
ax.title.set_text('t='+ str((T_in+5+step)*1.2)+'ms')
ax.set_ylabel('Solution')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(3,3,2)
pcm = ax.imshow(u_field[int(len(u_field)/2),:,:,-1], cmap='plasma', extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_2, vmax=v_max_2, aspect=aspect_ratio)
# ax.title.set_text('Middle')
ax.title.set_text('t='+ str((len(u_field)/2 +  T_in + 5 + step )*1.2)+'ms')
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(3,3,3)
pcm = ax.imshow(u_field[-1, :,:,-1], cmap='plasma',  extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_3, vmax=v_max_3, aspect=aspect_ratio)
# ax.title.set_text('Final')
ax.title.set_text('t='+ str((len(u_field)+  T_in + 5 + step )*1.2)+'ms')
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

# u_field = pred_set_reshaped[idx]
u_field = pred_set[int(np.sum(test_shot_lens[:shot_idx])):int(np.sum(test_shot_lens[:shot_idx+1])), :, :, :]

ax = fig.add_subplot(3,3,4)
pcm = ax.imshow(u_field[0,:,:,-1], cmap='plasma', extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_1, vmax=v_max_1, aspect=aspect_ratio)
ax.set_ylabel('FNO')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(3,3,5)
pcm = ax.imshow(u_field[int(len(u_field)/2),:,:,-1], cmap='plasma',  extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_2, vmax=v_max_2, aspect=aspect_ratio)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(3,3,6)
pcm = ax.imshow(u_field[-1,:,:,-1], cmap='plasma',  extent=[-1.5, 1.5, -2.0, 2.0], vmin=v_min_3, vmax=v_max_3, aspect=aspect_ratio)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

u_field = torch.abs(pred_set[int(np.sum(test_shot_lens[:shot_idx])):int(np.sum(test_shot_lens[:shot_idx+1])), :, :, :] - test_u[int(np.sum(test_shot_lens[:shot_idx])):int(np.sum(test_shot_lens[:shot_idx+1])), :, :, :])

ax = fig.add_subplot(3,3,7)
pcm = ax.imshow(u_field[0,:,:,-1], cmap='plasma', extent=[-1.5, 1.5, -2.0, 2.0], aspect=aspect_ratio)
ax.set_ylabel('Abs Error')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(3,3,8)
pcm = ax.imshow(u_field[int(len(u_field)/2),:,:,-1], cmap='plasma',  extent=[-1.5, 1.5, -2.0, 2.0], aspect=aspect_ratio)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

ax = fig.add_subplot(3,3,9)
pcm = ax.imshow(u_field[-1,:,:,-1], cmap='plasma',  extent=[-1.5, 1.5, -2.0, 2.0], aspect=aspect_ratio)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(pcm, cax=cax)
# cbar.formatter.set_powerlimits((0, 0))

plt.tight_layout()

plt.savefig(file_loc + '/rbb_last10_'+model_name+'_error.pdf')
# %%
#Testing out Discretisation Inference
# data_loc = os.path.dirname(data_loc) + '/DifferentResolutions_RBB/'
# shots_diff_resolutions = np.load(data_loc + 'shotnums_rbb_Different_Resolutions.npy')

# # %% 
# idx = np.random.randint(0, len(shots_diff_resolutions))
# idx = 10

# test_shots = shots_diff_resolutions[idx]
# test_a, test_u = time_windowing([test_shots])
# test_a = normalizer.encode(test_a)
# test_u_encoded = normalizer.encode(test_u)
# ntest = len(test_a)

# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u_encoded), batch_size=batch_size, shuffle=True)

# test_a.shape
# plt.imshow(test_a[112, :, :, 0])


# %%

# %%
#Creating Videos of the Plasma evolution. 

import imageio
import matplotlib.animation as animation

def create_video(vals, name):
  cmap = 'plasma'
  fps = 10 
  nSeconds = int(len(vals)/10)
  # First set up the figure, the axis, and the plot element we want to animate
  fig = plt.figure(figsize=(16,12))
  a = vals[0]
  im = plt.imshow(a, interpolation='none', aspect='auto', cmap=cmap)
  plt.axis('off')
  plt.tight_layout()
  plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)

  def animate_func(i):
      if i % fps == 0:
          print( '.', end ='' )

      im.set_array(vals[i])
      return [im]

  anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = nSeconds * fps,
                                interval = 1000 / fps, # in ms
                                )  

  output_path = file_loc + '/Videos/' + name + '.mp4'
  anim.save(output_path, fps=fps, extra_args=['-vcodec', 'libx264'])
  print('Done!')

# %% 
shot_idx = 3
camera_vals = test_u[int(np.sum(test_shot_lens[:shot_idx])):int(np.sum(test_shot_lens[:shot_idx+1])), :, :, 0]
fno_vals = pred_set[int(np.sum(test_shot_lens[:shot_idx])):int(np.sum(test_shot_lens[:shot_idx+1])), :, :, 0]

# %% 
# create_video(camera_vals, 'rbb_'+ str(test_shots[shot_idx]))
# create_video(fno_vals, 'fno_'+ model_name + '_' + str(test_shots[shot_idx]))
# create_video(torch.abs(fno_vals-camera_vals), 'error_'+ model_name + '_' + str(test_shots[shot_idx]))
