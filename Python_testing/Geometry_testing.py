# %%
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
#%%

Nx = 100; Ny = 100
A = np.zeros([Nx,Ny])
xa = np.linspace(-1,1,Nx)
ya = np.linspace(-1,1,Ny)
X,Y = np.meshgrid(xa,ya)
#print(X[:,0])
#print(Y)
for i in range(0,A[0].size,1):
    for j in range(0,A[1].size,1):
        #if (X[i,j]**2 + Y[i,j]**2) <= 0.25:
        #    A[i,j] = 3
        #elif (X[i,j]**2 + Y[i,j]**2) <= 0.5:
        #    A[i,j] = 2
        #elif (X[i,j]**2 + Y[i,j]**2) <= 0.75:
        #    A[i,j] = 1
        #else:
        #    A[i,j] = 0
         A[i,j] = X[i,j]**2 - Y[i,j]**2
        
#print(A)

fig = plt.figure()
axs = fig.add_subplot(111, projection='3d')
axs.plot_surface(X,Y,A, cmap=cm.inferno)
plt.show()
# %%

Sx = 1; Sy = 1
Nx = 100; Ny = 100

dx = Sx/Nx
xa = np.array([np.linspace(0,Nx,Nx)])*dx
xa = xa - xa.mean()

dy = Sy/Ny
ya = np.array([np.linspace(0,Ny,Ny)])*dy
ya = ya - ya.mean()

rx = 0.35
ry = 0.45
X,Y = np.meshgrid(xa,ya)
A = np.zeros([Nx,Ny])
for i in range(0,Nx,1):
     for j in range(0,Ny,1):
        if ((X[i,j]/rx)**2 + (Y[i,j]/ry)**2) <= 1:
            A[i,j] = 1
        else:
            A[i,j] = 0

plt.imshow(A)
plt.show()
fig = plt.figure()
axs = fig.add_subplot(111, projection='3d')
axs.plot_surface(X,Y,A, cmap=cm.inferno)
# %%
