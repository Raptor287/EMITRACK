import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.patches as patches

from mpl_toolkits.axes_grid1 import make_axes_locatable

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage


ExTime = np.load('EzTime_f.npy')
ParamStore = np.load('ParamStore_f.npy', allow_pickle=True)

X = ParamStore[0]; Y = ParamStore[1]; time_steps = ParamStore[2]; radius = ParamStore[3]

x = np.linspace(0,ExTime.shape[2]-1,ExTime.shape[2])

fps = 30
duration = time_steps/(fps*10)
fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax2 = fig.add_subplot(1,2,2)
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)

print(ExTime.shape)
def make_frame(anim_time):

    time_step = anim_time*fps

    ax1.cla()
    ax2.cla()

    ax1.dist = 9

    if (time_step <= 20):
        im = ax2.imshow(ExTime[int(time_step),:,:], cmap='viridis', norm=colors.CenteredNorm())
        surf = ax1.plot_surface(X,Y,ExTime[int(time_step),:,:].T, ccount = 100, rcount = 100, cmap=cm.jet, norm=colors.CenteredNorm())
    else: 
        im = ax2.imshow(ExTime[int(time_step),:,:], cmap='viridis', norm=colors.CenteredNorm())#, vmin=-1.5, vmax=1.5)
        surf = ax1.plot_surface(X,Y,ExTime[int(time_step),:,:].T, ccount =100, rcount = 100, cmap=cm.jet, norm=colors.CenteredNorm())
        ax1.set_zlim(-1.5, 1.5)
    device = patches.Rectangle((550, 0), width=2*radius, height=ExTime.shape[2], linewidth=1, edgecolor='w', facecolor='none')
    ax2.add_patch(device)
    ax2.set_title("Timestep = "+str(round(time_step*10)))
    fig.colorbar(im, cax=cax, orientation='vertical')

    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration = duration)

animation.ipython_display(fps = fps, loop = True, autoplay = True)
