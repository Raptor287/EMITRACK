# %%

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

Nx = 10
dx = 0.1
x = np.arange(0,Nx,dx)
y = np.sin(x)

# Building subplots in a figure
ax1 = plt.subplot(212)
ax1.plot(x,y)
# Annotating a particular subplot
ax1.annotate("Jason was here", [5,0.5])
ax2 = plt.subplot(221)
# Stem plots
ax2.stem(x,y)
ax3 = plt.subplot(222)
ax3.stem(y,x)
# plt.show() shows current figure
plt.show()

# Alternate way of doing above
fig, axs = plt.subplots(2,2)
axs[0,0].stem(x,y)
axs[0,1].stem(y,x)
axs[1,1].plot(x,y)

# Change color of plots
c = [1.0,0.8,0.0]
plt.setp(fig, facecolor = c)
plt.show()
# Print a given property of the figure
#print(fig.get_facecolor())

# %%
plota = np.zeros([50,50])
xa = np.linspace(-2,2,50)
ya = np.linspace(-1,1,25)
for i in range(0,xa.size,1):
    plota[:,i] = xa*i

# imshow/pcolor, pcolor gives control over individual cells
plt.imshow(plota.T)
plt.show()
plt.pcolor(plota.T, shading='auto')
plt.colorbar()
fig = plt.figure()
# 3D plotting
axs2 = fig.add_subplot(111, projection='3d')
# x-y meshgrid to plot function over
X,Y = np.meshgrid(xa,xa)
# Plotting the surface itself, cmap specifies color mapping
axs2.plot_surface(X,Y,plota, cmap=cm.inferno)
# %%
x = [0,1,1,0,0]
y = [0,0,1,1,0]
# plt.fill creates a polygon with the arguments as vertices
plt.figure()
square = plt.fill(x,y)
plt.show()

# Getting creative by using plt.fill to create a circle
phi = np.linspace(0,2*np.pi,20)
xc = np.cos(phi)
yc = np.sin(phi)
circle = plt.fill(xc,yc)
plt.show()
# %%
# importing movie py libraries
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
 
# numpy array
x = np.linspace(-2, 2, 200)
 
# duration of the video
duration = 2
 
# matplot subplot
fig, ax = plt.subplots()
 
# method to get frames
def make_frame(t):
     
    # clear
    ax.clear()
     
    # plotting line
    ax.plot(x, np.sinc(x**2) + np.sin(x + 2 * np.pi / duration * t), lw = 3)
    ax.set_ylim(-1.5, 2.5)
     
    # returning numpy image
    return mplfig_to_npimage(fig)
 
# creating animation
# --- Un-comment to create above movie ---
animation = VideoClip(make_frame, duration = duration)


# displaying animation with auto play and looping
animation.ipython_display(fps = 20, loop = True, autoplay = True)

# Interp 
# timing
# Use thin and thick lines to plot lines over busy backgrounds

# %%
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
plt.imshow(A)
plt.colorbar()
plt.show()
plt.pcolor(A)
plt.colorbar()
plt.show()
