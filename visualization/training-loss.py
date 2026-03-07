import matplotlib.pyplot as plt
import numpy as np

losses = np.load("experiments/losses.npy")

plt.plot(range(0, 2001, 10), losses)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.yscale("log")
min_idx = np.argmin(losses)
min_loss = losses[min_idx]
plt.axvline(min_idx*10,
            color="red",
            linestyle=":",
            linewidth=2,
            label=f"Min loss = {min_loss:.2e}")
plt.title("Training Loss")
plt.show()
min(losses)
