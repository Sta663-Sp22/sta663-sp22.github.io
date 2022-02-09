import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2, 201)
y = np.sin(2 * np.pi * x) + 1

plt.figure(layout="constrained")

plt.plot(x, y)
plt.title("About as simple as it gets, folks")
plt.xlabel("time (s)")
plt.ylabel("voltage (mV)")
plt.grid(True)

plt.show()
