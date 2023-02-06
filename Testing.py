# %%

import matplotlib.pyplot as plt
import numpy as np

Nx = 10
dx = 1
x = np.arange(0,Nx-1,dx)

y = np.sin(x)
plt.stem(x,y)
plt.show()
print(x[3])
# %%
