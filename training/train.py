import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from src.model import *
from src.losses import *
from src.exact_solution import *

torch.manual_seed(123)

pinn = FCN(
    N_INPUT=2,
    N_OUTPUT=1,
    N_HIDDEN=64,
    N_LAYERS=4
)

L = 1.0
T = 1.0

x_pde = -L + 2*L*torch.rand(1000,1)
t_pde = 1e-4 + (T-1e-4)*torch.rand(1000,1)
x_ic = torch.linspace(-L,L,200).view(-1,1)
t_bc = torch.rand(200,1)*T
x_mass = torch.linspace(-L,L,500).view(-1,1)
t_mass = torch.ones_like(x_mass)*0.5

# train the PINN
D = 0.02
Q = 1
sigma = 0.02

frames = []
x_test = -L + 2*L*torch.rand(1000,1)
t_test = 1e-4 + (T-1e-4)*torch.rand(1000,1)
optimiser = torch.optim.Adam(pinn.parameters(),lr=1e-3)
losses = []
x_plot = torch.linspace(-1,1,1000).view(-1,1)
t_plot = torch.ones_like(x_plot)*0.5
C_true =  Q/torch.sqrt(4*np.pi*D*t_plot) * torch.exp(-x_plot**2/(4*D*t_plot))

for i in range(1001):
    optimiser.zero_grad()

    l_pde  = loss_pde(pinn, x_pde, t_pde, D)
    l_ic   = loss_ic(pinn, x_ic, Q, sigma)
    l_bc   = loss_bc(pinn, t_bc, L)
    l_mass = loss_mass(pinn, x_mass, t_mass, Q)
   
    loss = l_pde + l_ic + l_bc + l_mass
    loss.backward()
    optimiser.step()
    if i % 10 == 0:
        with torch.no_grad():
            C_pred = pinn(x_plot,t_plot)
        frames.append(C_pred)
        losses.append(abs(C_pred - C_true).mean())
        
