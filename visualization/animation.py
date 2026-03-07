import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

frames = np.load("experiments/frames.npy")
losses = np.load("experiments/losses.npy")

fig, ax = plt.subplots()

line_pinn, = ax.plot([],[], '--', color="red", label="PINN")

line_true, = ax.plot((x_plot/torch.sqrt(2*D*t_plot)).numpy(),C_true,
                     color="black",linewidth=3,
                     label="Analytical")


ax.legend()
title = ax.set_title("")
ax.set_xlabel("x")
ax.set_ylabel("Concentration")

def update(i):
    line_pinn.set_data((x_plot/torch.sqrt(2*D*t_plot)).numpy(),frames[i])
    title.set_text(f"Iter {i} | Loss {losses[i].item():.2e}")
    return line_pinn, title

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=200,
    blit=True
)

ani.save("pinn_training.gif", writer="pillow", fps=5)
