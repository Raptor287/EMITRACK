#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

#%%

# Constants
e0 = 8.85418782e-12
u0 = 1.25663706e-6
z0 = np.sqrt(u0/e0)
c0 = 299792458

# Grid Params
wavelength = 1
dz = wavelength/40
Nz = int((wavelength*10)/dz)

# Device Params

# Pulse Params
tau = 0.5/300000000 # 0.5/fmax
t0 = 6*tau
nzpulse = 1 #int(Nz/2)
def pulse(t):
    return np.exp(-(((t*dt-t0)/tau)**2))

# Time Params
dt = dz/(2*c0)
tprop = (Nz*dz)/c0
time_steps = int(2*np.ceil((12*tau + 2*tprop)/dt))
cin = input("The calculated timesteps was "+str(time_steps)+". Would you like to use this? (y or new time_steps): ")
if cin != "y":
    time_steps = int(cin)
else: time_steps = int(time_steps)

# Update Coefficients
m_Ex = c0*dt # c0*dt/eps_xx
m_Hy = c0*dt # c0*dt/mu_yy

# Initializing Fields and Boundries
Ex = np.zeros((time_steps+1,Nz)); Hy = np.zeros((time_steps+1,Nz))
bound_low = [0,0]; bound_high = [0,0]

# Main FDTD Loop
for t in range(0,time_steps,1):
    if t%2 == 0:
        # Updating H-boundry (low)
        #bound_low[0] = bound_low[1]; bound_low[1] = Hy[t,0]
        #print(bound_low[0])
        # Hy field calculation
        for k in range(0,Nz-2,1):
            Hy[t+1,k] = Hy[t,k] + m_Hy*((Ex[t,k+1]-Ex[t,k])/dz)
        Hy[t+1,Nz-1] = Hy[t,Nz-1] + m_Hy*((bound_high[0] - Ex[t,Nz-1])/dz)

        # Update E-boundry 
        #bound_high[0] = bound_high[1]; bound_high[1] = Ex[t,Nz-1]
        # Ex field calculation
        Ex[t+1,0] = Ex[t,0]# + m_Ex*((Hy[t,0] - bound_low[0])/dz)
        for k in range(1,Nz-1,1):
            Ex[t+1,k] = Ex[t,k]# + m_Ex*((Hy[t,k] - Hy[t,k-1])/dz)
    
    else:
        # Updating H-boundry (low)
        #bound_low[0] = bound_low[1]; bound_low[1] = Hy[t,0]
        #print(bound_low[0])
        # Hy field calculation
        for k in range(0,Nz-2,1):
            Hy[t+1,k] = Hy[t,k] #+ m_Hy*((Ex[t,k+1]-Ex[t,k])/dz)
        Hy[t+1,Nz-1] = Hy[t,Nz-1] #+ m_Hy*((bound_high[0] - Ex[t,Nz-1])/dz)

        # Update E-boundry 
        #bound_high[0] = bound_high[1]; bound_high[1] = Ex[t,Nz-1]
        # Ex field calculation
        Ex[t+1,0] = Ex[t,0] + m_Ex*((Hy[t,0] - bound_low[0])/dz)
        for k in range(1,Nz-1,1):
            Ex[t+1,k] = Ex[t,k] + m_Ex*((Hy[t,k] - Hy[t,k-1])/dz)
        
    
    # Inject source
    Ex[t+1,nzpulse-1] = Ex[t+1,nzpulse-1] + pulse(t+1)
    print(Ex[t,nzpulse-1])
#%%

z = np.arange(0,Nz*dz,dz)

fps = 60
duration = time_steps/fps
fig, ax = plt.subplots()

def make_frame(anim_time):

    time_step = anim_time*fps
    ax.clear()

    ax.plot(z, Ex[int(time_step),:])
    ax.set_title("Timestep = "+str(round(time_step)))
    ax.set_ylim(-1,10)

    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration = duration)

animation.ipython_display(fps = fps, loop = True, autoplay = True)


#%%

#t = np.arange(0,time_steps,1)
#plt.plot(t,pulse(t))
#plt.show()