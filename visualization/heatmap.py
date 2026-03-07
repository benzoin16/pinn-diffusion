from matplotlib import colors
import matplotlib.pyplot as plt

import torch


x = torch.linspace(-1,1,100)
t = torch.linspace(0.01,1,100)

X,T = torch.meshgrid(x,t,indexing='ij')

Xf = X.reshape(-1,1)
Tf = T.reshape(-1,1)

with torch.no_grad():
    C = pinn(Xf,Tf)

C = C.reshape(100,100)

plt.figure()

plt.imshow(C,
           extent=[0,1,-1,1],
           aspect="auto",
           origin="lower",
           norm=colors.LogNorm())

plt.colorbar(label="Concentration")

plt.xlabel("Time")
plt.ylabel("Position")

plt.title("Diffusion Evolution")

plt.show()
