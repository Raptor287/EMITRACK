#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

#%%

# Timer to measure program execution time
timer_start = time.time()

# Constants
e0 = 8.85418782e-12
u0 = 1.25663706e-6
z0 = np.sqrt(u0/e0)
c0 = 299792458

# Grid Params
wavelength = 1
dz = wavelength/40
Nz = int((wavelength*10)/dz)

# Device Params (no device yet)


# Time Params
# dt such that the wave propogates one cell in 2 time steps; tprop is approximate propogation time across the grid.
dt = dz/(2*c0)
tprop = (Nz*dz)/c0

# Pulse Params
tau = 0.5/300000000                                 # 0.5/fmax, spread of pulse
t0 = 6*tau                                          # Time offset to ease into pulse
nzpulse = int(Nz/4) - 1                             # Location of pulse, array index friendly
H_scale = 1                                         # sqrt(e_rel/u_rel), normalization of 'Hy' source due to derivation of update Eqs
H_offset = dz/(2*c0) - 0.5*dt                       # (n_source*dz)/(2*c0) - (delta_t/2), 'Hy' source offset due to time/grid offset
                                                        # There is likely an issue here. The offset works, but I cannot explain it
def pulse(t,Offset):
    return np.exp(-(((t*dt-t0+Offset)/tau)**2))

# Iteration Number Calculation
time_steps = int(np.ceil((12*tau + 3*tprop)/dt))  # time_steps is iterations to ease into and out of source and propogate 3 times.
cin = input("The calculated timesteps was "+str(time_steps)+". Would you like to use this? (y or new time_steps): ")
if cin != "y":
    time_steps = int(cin)
else: time_steps = int(time_steps)

# Update Coefficients
m_Ex = c0*dt/1 # c0*dt/eps_xx
m_Hy = c0*dt/1 # c0*dt/mu_yy
# note eps_xx and mu_yy are relative to e0 and u0
# also note, it might be possible to include dz in this calculation

# Initializing Fields and Boundries
Ex = np.zeros((time_steps+1,Nz)); Hy = np.zeros((time_steps+1,Nz))
bound_low = [0,0]; bound_high = [0,0]

# Main FDTD Loop
for t in range(0,time_steps,1):
    # Magnetic Field Update (t+dt/2) 
    for k in range(0,Nz-1,1):
         Hy[t+1,k] = Hy[t,k] - m_Hy*((Ex[t,k+1]-Ex[t,k])/dz)                # Standard Hy update
    Hy[t+1,Nz-1] = Hy[t,Nz-1] - m_Hy*((bound_high[0] - Ex[t,Nz-1])/dz)      # Hy update at end of grid
    Hy[t+1,nzpulse-1] = Hy[t+1,nzpulse-1] + m_Hy*(pulse(t,0)/dz)            # TF/SF pulse. Pulse is subtracted from Ex[t,nzpulse]. Note the 't-2'.
    # Lower Boundry Update (t-dt/2)
    #bound_low[0] = bound_low[1]; bound_low[1] = Hy[t,0]

    # Ex Fied Update (t+dt)
    Ex[t+1,0] = Ex[t,0] - m_Ex*((Hy[t+1,0] - bound_low[0])/dz)                  # Ex update at beginning of grid
    for k in range(1,Nz,1):
        Ex[t+1,k] = Ex[t,k] - m_Ex*((Hy[t+1,k] - Hy[t+1,k-1])/dz)               # Standard update
    Ex[t+1,nzpulse] = Ex[t+1,nzpulse] + m_Ex*(H_scale*pulse(t+1,H_offset)/dz)   # TF/SF pulse. Pulse is added to Hy[t,nzpulse-1]
    # Upper Boundry Update (t)
    #bound_high[0] = bound_high[1]; bound_high[1] = Ex[t,Nz-1]

timer_end = time.time()
with open("Samples/Timer.txt", "a") as f:
    print("Execution on", time.strftime("%d %b %Y at %H:%M:%S took", time.localtime()), timer_end - timer_start, "seconds\n", file=f)
#%%

# A potential data output method. This plots the magnitude of E at a constant position over time
t = np.arange(0,time_steps+1,1)
plt.plot(t/dt,Ex[:,nzpulse+1])
plt.title("|E| Over Time at Source Position + 1")
plt.xlabel("Time")
plt.ylabel("|E|")
plt.savefig(fname="E_Over_Time_TFSF_Source")
plt.show()

#%%
z = np.arange(0,Nz*dz,dz)
fps = 30
duration = time_steps/(fps*10)
fig, ax = plt.subplots()

def make_frame(anim_time):

    time_step = anim_time*fps*10
    ax.clear()

    ax.plot(z, Ex[int(time_step),:], label='E-Field')
    ax.plot(z, Hy[int(time_step),:], label='H-Field')
    ax.legend()
    ax.set_title("Timestep = "+str(round(time_step)))
    ax.set_ylim(-5,5)

    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration = duration)

animation.ipython_display(fps = fps, loop = True, autoplay = True)


#%%
# Fast Fourier Transform for Gaussian Pulse
'''
t = np.arange(0,time_steps,1)
g = pulse(t)
plt.plot(t,g)
plt.xlabel('Time Step')
plt.ylabel('Amplitude')
plt.title('Gaussian Pulse - Time Step Domain')

#%%
G = np.fft.rfft(g)
f = np.arange(0,time_steps/2+1)*((1/dt)/time_steps)
plt.plot(f,np.abs(G))
plt.xlim(0,1e9)
plt.xlabel('Frequency (1e9)')
plt.ylabel('Power')
plt.title('Gaussian Pulse - Frequency Domain')
#plt.plot(t,pulse(t))
#plt.show()
'''
# %%
