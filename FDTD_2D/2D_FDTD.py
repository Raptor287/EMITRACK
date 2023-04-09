#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

#%%

# Timer to measure program Eyecution time
timer_start = time.time()

# Constants
e0 = 8.85418782e-12
u0 = 1.25663706e-6
z0 = np.sqrt(u0/e0)
c0 = 299792458
ur_max = 1.0
er_max = 1.0


# Computing Grid Resolution

## Wavelength Resolution
f_max = 1e8                                         # 100 MHz
n_max = np.sqrt(ur_max*er_max)                      # Max refractive indEy in grid
wavelength_min = c0/(f_max*n_max)                   # Min wavelength in grid
N_wave = 20                                         # Wave resolution
delta_wave = wavelength_min/N_wave                  # Grid resolution from wavelength

## Structure Resolution
#            # Grid resolution from structure

## Initial Grid Resolution
dz = min(delta_wave,delta_device)                   # Initial res is smaller of wave/struct res

## Snapping Grid to Critical Dimension of Device
N_device = int(np.ceil(structure_min/dz))           # Number of cells to represent structure is now an integer
dz = structure_min/N_device                         # dz recalculated with N_device


# Building Grid

## Determining Grid Size
Nz = N_device + 2*500 + 3                            # N_device cells for device, 1000 spacer cells, and 3 source/record cells

## Building device
Device_start = 2 + 500                               # 2 cells for source/record, 500 spacers
Device_end = Device_start + N_device                # End of device indEy
mu_xx = np.ones(Nz)
mu_xx[Device_start:Device_end] = ur_max             # Array mu_xx contains permeablility across grid
eps_yy = np.ones(Nz)
eps_yy[Device_start:Device_end] = er_max            # Permittivity across grid
# Note: This is the device. Its simply represented by mu and eps values across the grid


# Time Params

dt = dz/(2*c0)                                      # dt such that the wave propogates one cell in 2 time steps (n_bounds*dz/(2*c0))
tprop = (n_max*Nz*dz)/c0                            # tprop is the approximate time for propogation across the grid


# Pulse Params
tau = 0.5/f_max                                     # 0.5/fmax, spread of pulse
t0 = 6*tau                                          # Time offset to ease into pulse


# Iteration Number Calculation
sim_time = 12*tau + 2*tprop                         # Total simulation time
time_steps = int(np.ceil((sim_time)/dt))            # time_steps is iterations required to ease into and out of source and propogate accross the grid 5 times.
cin = input("The calculated timesteps was "+str(time_steps)+". Would you like to use this? (y or new time_steps): ")
if cin != "y":
    time_steps = int(cin)
else: time_steps = int(time_steps)


# Source Functions (TF/SF)
t_sec = np.array(np.arange(0,time_steps+1,1))*dt    # Time array in seconds
k_source = 1                                        # Location of pulse, array indEy friendly
H_offset = dz/(2*c0) + 0.5*dt                       # (n_source*dz)/(2*c0) + (delta_t/2), 'Hx' source offset due to time/grid offset    
H_scale = -np.sqrt(eps_yy[k_source]/mu_xx[k_source])    # -sqrt(e_rel/u_rel), normalization of 'Hx' source due to derivation of update Eqs
E_source = np.exp(-(((t_sec-t0)/tau)**2))               # Electric field source
H_source = H_scale*np.exp(-(((t_sec-t0+H_offset)/tau)**2)) # Magnetic field source


# Update Coefficients
m_Ey = c0*dt/eps_yy                                 # c0*dt/eps_yy
m_Hx = c0*dt/mu_xx                                  # c0*dt/mu_xx
# Note: eps_yy and mu_xx are relative to e0 and u0
# Also note: it might be possible to include dz in this calculation


# Initializing Fields and Boundries
Ey = np.zeros((time_steps+1,Nz)); Hx = np.zeros((time_steps+1,Nz))
bound_low = [0,0]; bound_high = [0,0]


# Main FDTD Loop
for t in range(0,time_steps,1):

    # Magnetic Field Update in x dir. Note: Hx[t+1] = Hx(t+dt/2) 
    for k in range(0,Nz-1,1):
         Hx[t+1,k] = Hx[t,k] + m_Hx[k]*((Ey[t,k+1]-Ey[t,k])/dz)                    # Standard Hx update
    Hx[t+1,Nz-1] = Hx[t,Nz-1] + m_Hx[k]*((bound_high[0] - Ey[t,Nz-1])/dz)          # Hx update at end of grid
    Hx[t+1,k_source-1] = Hx[t+1,k_source-1] - m_Hx[k_source]*(E_source[t]/dz)                # TF/SF pulse. Pulse is subtracted from Ey[t,k_source].
    # Lower Boundry Update (t-dt/2)
    bound_low[0] = bound_low[1]; bound_low[1] = Hx[t,0]

    # Electric Fied Update in x dir. Note: Ey[t+1] = Ey(t+dt)
    Ey[t+1,0] = Ey[t,0] + m_Ey[k]*((Hx[t+1,0] - bound_low[0])/dz)                  # Ey update at beginning of grid
    for k in range(1,Nz,1):
        Ey[t+1,k] = Ey[t,k] + m_Ey[k]*((Hx[t+1,k] - Hx[t+1,k-1])/dz)               # Standard update
    Ey[t+1,k_source] = Ey[t+1,k_source] - m_Ey[k_source]*(H_source[t]/dz)          # TF/SF pulse. Pulse is added to Hx[t+1,k_source-1]. Note: 'Hx[t+1,:]' is 'Hx' evaluated at 't+dt/2'
    # Upper Boundry Update (t)
    bound_high[0] = bound_high[1]; bound_high[1] = Ey[t,-1]


timer_end = time.time()
print("Program took:", timer_end-timer_start, "seconds.")

# Timer logging below as required for analysis 
#with open("Samples/Timer.txt", "a") as f:
#    print("Execution on", time.strftime("%d %b %Y at %H:%M:%S took", time.localtime()), timer_end - timer_start, "seconds.", file=f)
#%%

# A potential data output method. This plots the magnitude of E at a constant position over time
plt.plot(t_sec,Ey[:,0], label='Reflected')
plt.plot(t_sec,Ey[:,-1], label='Transmitted')
plt.legend()
plt.title("|E| Over Time at Boundries")
plt.xlabel("Time")
plt.ylabel("|E|")
plt.savefig(fname="Ebound_Over_Time_TFSF_Source")
plt.show()

# Another data output method that calculates the frequencies transmitted and reflected by doing an FFT of Ey at boundries of grid and normalizing to the source
Freq = np.arange(0,time_steps/2+1)*((1/dt)/time_steps)
SRC = abs(np.fft.rfft(E_source[:]))
REF = abs(np.fft.rfft(Ey[:,0])); REF = (REF/SRC)**2
TRN = abs(np.fft.rfft(Ey[:,-1])); TRN = (TRN/SRC)**2
TOT_TR = REF+TRN

plt.title("Frequency Reflectance and Transmittance")
plt.plot(Freq,REF*100, label='REF')
plt.plot(Freq,TRN*100, label='TRN')
plt.plot(Freq,TOT_TR*100, label='TOT_TR')
plt.xlim(0,f_max)
plt.ylim(0,150)
plt.legend(loc='upper left')
plt.savefig(fname="Freq_T_and_R")
plt.show()

#%%

z = np.arange(0,Nz*dz,dz)
device_x = [Device_start*dz,Device_start*dz,Device_end*dz,Device_end*dz]
device_y = [-2,2,2,-2]
fps = 30
duration = time_steps/(fps*10)
fig, ax = plt.subplots()

def make_frame(anim_time):

    time_step = anim_time*fps*10

    ax.clear()
    ax.fill(device_x,device_y, alpha=0.5)
    ax.plot(z, Ey[int(time_step),:], label='E-Field')
    ax.plot(z, Hx[int(time_step),:], label='H-Field')
    ax.legend()
    ax.set_title("Timestep = "+str(round(time_step)))
    ax.set_ylim(-3,3)

    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration = duration)

animation.ipython_display(fps = fps, loop = True, autoplay = True)
