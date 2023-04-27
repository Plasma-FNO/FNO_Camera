#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 6 Jan 2023
@author: vgopakum
Exploring the Camera Dataset - rbb and rba 
"""
# %%
#Importing the necessary packages. 
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

from mpl_toolkits.axes_grid1 import make_axes_locatable

# %% 
import os 
path = os.getcwd()
data_loc = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
# model_loc = os.path.dirname(os.path.dirname(os.getcwd()))
file_loc = os.getcwd()


# %% 
def plot(val, idx):
    u_field = val[idx]

    v_min_1 = torch.min(u_field[:,:,0])
    v_max_1 = torch.max(u_field[:,:,0])

    v_min_2 = torch.min(u_field[:, :, int(T/2)])
    v_max_2 = torch.max(u_field[:, :, int(T/2)])

    v_min_3 = torch.min(u_field[:, :, -1])
    v_max_3 = torch.max(u_field[:, :, -1])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1,3,1)
    pcm =ax.imshow(u_field[:,:,0], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_1, vmax=v_max_1)
    # ax.title.set_text('Initial')
    ax.title.set_text('t='+ str(0))
    ax.set_ylabel(idx)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))
    
    ax = fig.add_subplot(1,3,2)
    pcm = ax.imshow(u_field[:,:,int(T/2)], cmap=cm.coolwarm, extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_2, vmax=v_max_2)
    # ax.title.set_text('Middle')
    ax.title.set_text('t='+ str(int((T)/2)))
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))

    ax = fig.add_subplot(1,3,3)
    pcm = ax.imshow(u_field[:,:,-1], cmap=cm.coolwarm,  extent=[9.5, 10.5, -0.5, 0.5], vmin=v_min_3, vmax=v_max_3)
    # ax.title.set_text('Final')
    ax.title.set_text('t='+str(T))
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.formatter.set_powerlimits((0, 0))

# %% 
#Exploring RBB dataset

#  30055 - 30430 : Initial RBB Camera Data
#  29920 - 29970 : moved RBB Camera Data

case = 'RBB Camera'
if case == 'RBB Camera':

    data =  np.load(data_loc + '/Data/Cam_Data/Cleaned_Data/rbb_30055_30430.npy')
    # data_calib =  np.load(data_loc + '/Data/Cam_Data/Cleaned_Data/Calibrations/rbb_rz_pos_30055_30430.npz')

elif case == 'RBB Camera - Moved':

    data =  np.load(data_loc + '/Data/Cam_Data/Cleaned_Data/rbb_29920_29970.npy')
    # data_calib =  np.load(data_loc + '/Data/Cam_Data/Cleaned_Data/Calibrations/rbb_rz_pos_29920_29970.npz')

# %% 

u_sol = data.astype(np.float32)

grid_size_x = u_sol.shape[2]
grid_size_y = u_sol.shape[3]
S_x = grid_size_x #Grid Size
S_y = grid_size_y #Grid Size

u = torch.from_numpy(u_sol)
u = u.permute(0, 2, 3, 1)
u = u[...,5:75]
T = 60


# %%
plot(u, 0)
# %%
for ii in range(len(u)):
    plot(u, ii)
# %%
rbb_normal_idx = [5, 30, 34, 48, 52]

# %%
#Exploring the RBA Dataset 

data =  np.load(data_loc + '/Data/Cam_Data/Cleaned_Data/rba_30280_30360.npy')
# data_2 = np.load(data_loc + '/Data/Cam_Data/rba_fno_data_2.npy')
# data =  np.load(data_loc + '/Data/Cam_Data/rba_data_608x768.npy')
# data_calib =  np.load(data_loc + '/Data/Cam_Data/Cleaned_Data/Calibrations/rba_rz_pos_30280_30360.npz')

u_sol = data.astype(np.float32)

# u_2_sol = data_2.astype(np.float32)
# u_sol = np.vstack((u_sol, u_2_sol))


# %%

grid_size_x = u_sol.shape[2]
grid_size_y = u_sol.shape[3]

u = torch.from_numpy(u_sol)
u = u.permute(0, 2, 3, 1)
u = u[...,5:75]
T = 60
# %%
for ii in range(len(u)):
    plot(u, ii)
# %%
rba_1_idx = [25, 40]

#Also start the training from the first 5 onwards. 