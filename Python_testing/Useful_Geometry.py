# %%
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
#%%

# 3D Saddle
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

plt.figure()
cplot = plt.contour(X,Y,A)
plt.clabel(cplot, cplot.levels, inline=True)
plt.show()
# %%

# Elliptical Cylinder

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

plt.figure()
plt.imshow(A)
plt.show()
fig = plt.figure()
axs = fig.add_subplot(111, projection='3d')
axs.plot_surface(X,Y,A, cmap=cm.inferno)
plt.show()
# %%

# Formed Half-Spaces

Sx = 1; Sy = 1
Nx = 100; Ny = 100

dx = Sx/Nx
xa = np.array([np.linspace(0,Nx,Nx)])*dx
xa = xa - xa.mean()

dy = Sy/Ny
ya = np.array([np.linspace(0,Ny,Ny)])*dy
ya = ya - ya.mean()

y = 0.2 + 0.1*np.cos(4*np.pi*xa/Sx)

A = np.zeros([Nx,Ny])
for i in range(0,Nx,1):
    j = round((y[0,i] + Sy/2)/dy)
    A[i,:j] = 1

plt.figure()
plt.imshow(A.T, origin='lower')
plt.colorbar()
plt.show()
# %%

# Linear Half-Spaces

Sx = 1; Sy = 1
Nx = 100; Ny = 100

dx = Sx/Nx
xa = np.array([np.linspace(0,Nx,Nx)])*dx
xa = xa - xa.mean()

dy = Sy/Ny
ya = np.array([np.linspace(0,Ny,Ny)])*dy
ya = ya - ya.mean()

X,Y = np.meshgrid(xa,ya)

x1 = -0.5; y1 = 0.25
x2 = 0.5; y2 = -0.25

m = (y2-y1)/(x2-x1)
A = np.zeros([Nx,Ny])
for i in range(0,Nx,1):
    for j in range(0,Ny,1):
        if (Y[i,j]-y1) - m*(X[i,j]-x1) > 0:
            A[i,j] = 1

plt.figure()
plt.imshow(A, origin='lower')
plt.colorbar()
plt.show()
# %%

# Combined Spaces

Sx = 1; Sy = 1
Nx = 100; Ny = 100

dx = Sx/Nx
xa = np.array([np.linspace(0,Nx,Nx)])*dx
xa = xa - xa.mean()

dy = Sy/Ny
ya = np.array([np.linspace(0,Ny,Ny)])*dy
ya = ya - ya.mean()

X,Y = np.meshgrid(xa,ya)

y = -0.2 + 0.1*np.cos(4*np.pi*xa/Sx)

FS = np.zeros([Nx,Ny])
for i in range(0,Nx,1):
    j = round((y[0,i] + Sy/2)/dy)
    FS[i,j:] = 1

x1 = -0.5; y1 = 0.5
x2 = 0.5; y2 = -0.5

m = (y2-y1)/(x2-x1)
LHS = np.zeros([Nx,Ny])
for i in range(0,Nx,1):
    for j in range(0,Ny,1):
        if (Y[i,j]-y1) - m*(X[i,j]-x1) > 0:
            LHS[i,j] = 1

A = FS*LHS
plt.figure()
plt.imshow(A.T, origin='lower')
plt.colorbar()
plt.show()

# %%

# Scaling Values of the Grids

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

er1 = 1.0; er2 = 2.4
A = er1 + (er2 - er1)*A

plt.figure()
plt.imshow(A, origin='lower')
plt.colorbar()
plt.show()
fig = plt.figure()
axs = fig.add_subplot(111, projection='3d')
axs.plot_surface(X,Y,A, cmap=cm.inferno)
plt.show()
