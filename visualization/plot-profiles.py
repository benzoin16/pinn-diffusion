import matplotlib.pyplot as plt
import torch
from src.model import *


times = [0.1,0.3,0.6,1.0]
pinn = FCN(
    N_INPUT=2,
    N_OUTPUT=1,
    N_HIDDEN=64,
    N_LAYERS=4
)
plt.figure()

for t_val in times:

    t_plot = torch.ones_like(x_plot)*t_val

    with torch.no_grad():
        C_pred = pinn(x_plot,t_plot)

    plt.plot(x_plot.numpy(),C_pred.numpy(),label=f"t={t_val}")

plt.legend()
plt.xlabel("x")
plt.ylabel("C")
plt.title("Diffusion over time")
plt.show()
