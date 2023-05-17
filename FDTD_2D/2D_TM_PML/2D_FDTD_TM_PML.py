#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import time

from numba import jit

from mpl_toolkits.axes_grid1 import make_axes_locatable

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

#%%

# Timer to measure program Execution time
timer_start = time.time()
timer_delay = [0,0,0]                               # delay to compensate for user input time. First 2 indexes are temp, 3rd stores total delay

# Constants
e0 = 8.85418782e-12
u0 = 1.25663706e-6
z0 = np.sqrt(u0/e0)
c0 = 299792458
ur_max = 1.0
er_max = 1.0

# Radar Params
f0 = 32.55e7                                        # Radar central freq, 32.55 MHz
pulse_width = 0.1e-7                                # Width of radar pulse in seconds (normally 5 us, set to 0.01 us for time sake)

# Computing Grid Resolution

## Wavelength Resolution
f_max = 50e7                                        # Can resolve up to 50 MHz
n_max = np.sqrt(ur_max*er_max)                      # Max refractive index in grid
wavelength_min = c0/(f_max*n_max)                   # Min wavelength in grid
N_wave = 20                                         # Wave resolution
delta_wave = wavelength_min/N_wave                  # Grid resolution from wavelength
dx = dy = delta_wave                                


# Building Grid

## Selecting Grid Size
### Nx2 and Ny2 are 2x grids that are used only to model the device. 'mu' and 'eps' will be pulled from 2x grid
Nx = 500; Nx2 = 2*Nx
Ny = 1000; Ny2 = 2*Ny

## Building Grid
xa = np.linspace(0,Nx-1,Nx)
ya = np.linspace(0,Ny-1,Ny)
### Lengths of the PML in the x and y directions
PML_Lx = np.array([20,20])
PML_Ly = np.array([20,20])
X,Y = np.meshgrid(xa,ya)

## Building Device
### X and Y position of cylinder in number of cells
Device_x = 250
Device_y = 550
### Cylinder radius in number of cells
Device_radius = int(np.ceil((c0/(2*np.pi*f0))/dx))           # Note the radius in meters is divided by dx to get the radius in cells rounded up
### Setting points inside cylinder radius to be perfect conductors
Pec = np.ones((Nx2,Ny2))
for i in range(0,Nx2,1):
    for j in range(0,Ny2,1):
        if ((i-Device_x)**2+(j-Device_y)**2 <= Device_radius**2):
            Pec[i,j] = 0

### Mu
mu_xx = np.ones((Nx,Ny))
mu_yy = np.ones((Nx,Ny))

### Eps
eps_zz = np.ones((Nx,Ny))


# Time Params

dt = dx/(2*c0)                                      # dt such that the wave propogates one cell in 2 time steps (n_bounds*dz/(2*c0))
tprop = (n_max*Nx*dx)/c0                            # tprop is the approximate time for propogation across the grid


# Pulse Params

tau = 2/f_max                                       # 4*0.5/fmax, spread of gaussian profile to ease into pulse
t0 = 6*tau                                          # Time offset to ease into pulse


# Iteration Number Calculation

sim_time = 12*tau + 5*tprop + pulse_width           # Total simulation time
time_steps = int(np.ceil((sim_time)/dt))            # time_steps is iterations required to ease into and out of source and propogate accross the grid 5 times.
timer_delay[0] = time.time()
cin = input("The calculated timesteps was "+str(time_steps)+". Would you like to use this? (y or new time_steps): ")
if cin != "y":
    time_steps = int(cin)
else: time_steps = int(time_steps)
timer_delay[1] = time.time()
timer_delay[2] = timer_delay[1] - timer_delay[0]


# Source Functions (TF/SF)

t_sec = np.array(np.arange(0,time_steps+1,1))*dt    # Time array in seconds
j_source = PML_Ly[0] + 3                            # Location of pulse, array index friendly
H_offset = dy/(2*c0) + 0.5*dt                       # (n_source*dz)/(2*c0) + (delta_t/2), 'Hx' source offset due to time/grid offset    
H_scale = -np.sqrt(eps_zz[int(Nx/2),j_source]/mu_xx[int(Nx/2),j_source])    # -sqrt(e_rel/u_rel), normalization of 'Hx' source due to derivation of update Eqs
## Pulse shaping using gaussian to ease in/out of source
Ez_source_shape = np.exp(-(((t_sec-t0)/tau)**2))
Hx_source_shape = np.exp(-(((t_sec-t0+H_offset)/tau)**2))
for t in range(0,time_steps+1,1):
    if t*dt > t0 and t*dt < t0+pulse_width:
        Ez_source_shape[t] = 1
        Hx_source_shape[t] = 1
    elif t*dt >= t0+pulse_width:
        Ez_source_shape[t] = np.exp(-(((t_sec[t]-t0-pulse_width)/tau)**2))
        Hx_source_shape[t] = np.exp(-(((t_sec[t]-t0-pulse_width+H_offset)/tau)**2))

## Source functions
Ez_source = Ez_source_shape*np.cos(2*np.pi*f0*t_sec)          # Electric field source
Hx_source = Hx_source_shape*H_scale*np.cos(2*np.pi*f0*(t_sec+H_offset)) # Magnetic field source


# PML Parameters

## Sigma x
sig_x = np.zeros((Nx2,Ny2))
for i in range(0,2*PML_Lx[0]):
    sig_x[2*PML_Lx[0]-i-1,:] = (0.5*e0/dt)*(i/(2*PML_Lx[0]))**3
for i in range(0,2*PML_Lx[1]):
    sig_x[Nx2-2*PML_Lx[1]+i,:] = (0.5*e0/dt)*(i/(2*PML_Lx[1]))**3

## Sigma y
sig_y = np.zeros((Nx2,Ny2))
for j in range(0,2*PML_Ly[0]):
    sig_y[:,2*PML_Ly[0]-j-1] = (0.5*e0/dt)*(j/(2*PML_Ly[0]))**3
for j in range(0,2*PML_Ly[1]):
    sig_y[:,Ny2-2*PML_Ly[1]+j] = (0.5*e0/dt)*(j/(2*PML_Ly[1]))**3


# Update Coefficients

## Hx Coefficients
sigHx = sig_x[0::2,1::2]; sigHy = sig_y[0::2,1::2]
m_Hx0 = (1/dt) + sigHy/(2*e0)
m_Hx1 = ((1/dt) - sigHy/(2*e0))/m_Hx0
m_Hx2 = - c0/(mu_xx*m_Hx0)
m_Hx3 = - (c0*dt/e0) * sigHx/(mu_xx*m_Hx0)

## Hy Coefficients
sigHx = sig_x[1::2,0::2]; sigHy = sig_y[1::2,0::2]
m_Hy0 = (1/dt) + sigHx/(2*e0)
m_Hy1 = ((1/dt) - sigHx/(2*e0))/m_Hy0
m_Hy2 = - c0/(mu_yy*m_Hy0)
m_Hy3 = - (c0*dt/e0) * sigHy/(mu_yy*m_Hy0)

## Dz Coefficients
sigDx = sig_x[0::2,0::2]; sigDy = sig_y[0::2,0::2]
m_Dz0 = (1/dt) + ((sigDx + sigDy)/(2*e0)) + (sigDx*sigDy*dt/(4*(e0**2)))
m_Dz1 = (1/dt) - ((sigDx + sigDy)/(2*e0)) - (sigDx*sigDy*dt/(4*(e0**2))); m_Dz1 = m_Dz1/m_Dz0
m_Dz2 = c0/m_Dz0
m_Dz4 = - (dt/(e0**2))*sigDx*sigDy/m_Dz0

# Note: mu_xx and mu_yy are relative to u0


# Initializing Fields and Boundries

## Magnetic Fields
### The time arrays are very large (20Gb with 5000x5000 grid and 1000 steps). Use sparingly.
### 'dtype = np.half' uses half precision float to save memory (5gb with above params)
#HxTime = np.zeros((int(time_steps/10)+1,Nx,Ny),dtype=np.half); HyTime = np.zeros((int(time_steps/10+1),Nx,Ny),dtype=np.half); 
Hx = np.zeros((Nx,Ny)); Hy = np.zeros((Nx,Ny))
I_CEx = np.zeros((Nx,Ny)); I_CEy = np.zeros((Nx,Ny))
CHz = np.zeros((Nx,Ny))

## Electric Fields
EzTime = np.zeros((int(time_steps/10)+1,Nx,Ny),dtype=np.single)
EzReciever = np.zeros(time_steps+1)
Ez = np.zeros((Nx,Ny)); Dz = np.zeros((Nx,Ny))
I_Dz = np.zeros((Nx,Ny))
CEx = np.zeros((Nx,Ny)); CEy = np.zeros((Nx,Ny))


# Main FDTD Loop
@jit(nopython=True)
def FDTD_Loop(EzTime,EzReciever,Ez,Dz,CEx,CEy,I_Dz,Hx,Hy,CHz,I_CEx,I_CEy):
    for t in range(0,time_steps+1,1):

        # Magnetic Field Update

        ## Ex and Ey Curl Updates
        for i in range(0,Nx,1):
            for j in range(0,Ny-1,1):
                CEx[i,j] = (Ez[i,j+1] - Ez[i,j])/dy
            CEx[i,Ny-1] = (0 - Ez[i,Ny-1])/dy
        
        for i in range(0,Nx,1):
            CEx[i,j_source-1] = CEx[i,j_source-1] - Ez_source[t]/dy
        
        for j in range(0,Ny,1):
            for i in range(0,Nx-1,1):
                CEy[i,j] = - (Ez[i+1,j] - Ez[i,j])/dx
            CEy[Nx-1,j] = - (0 - Ez[Nx-1,j])/dx

        ## Ex and Ey Integrations
        I_CEx = I_CEx + CEx
        I_CEy = I_CEy + CEy

        ## Hx and Hy Updates
        for i in range(0,Nx,1):
            for j in range(0,Ny,1):
                Hx[i,j] = m_Hx1[i,j]*Hx[i,j] + m_Hx2[i,j]*CEx[i,j] + m_Hx3[i,j]*I_CEx[i,j]
                Hy[i,j] = m_Hy1[i,j]*Hy[i,j] + m_Hy2[i,j]*CEy[i,j] + m_Hy3[i,j]*I_CEy[i,j]
        
        # Electric Field Update

        ## Hz Curl Update
        ### 0 Corner Update
        CHz[0,0] = (Hy[0,0] - 0)/dx - (Hx[0,0] - 0)/dy
        ### y=0 Row Update
        for i in range(1,Nx,1):
            CHz[i,0] = (Hy[i,0] - Hy[i-1,0])/dx - (Hx[i,0] - 0)/dy
        ### Remaining Grid Update
        for j in range(1,Ny,1):
            CHz[0,j] = (Hy[0,j] - 0)/dx - (Hx[0,j] - Hx[0,j-1])/dy
            for i in range(1,Nx,1):
                CHz[i,j] = (Hy[i,j] - Hy[i-1,j])/dx - (Hx[i,j] - Hx[i,j-1])/dy
        
        for i in range(0,Nx,1):
            CHz[i,j_source] = CHz[i,j_source] - Hx_source[t]/dy
        
        ## Dz Integration
        I_Dz = I_Dz + Dz

        ## Dz Field Update
        for i in range(0,Nx,1):
            for j in range(0,Ny,1):
                Dz[i,j] = m_Dz1[i,j]*Dz[i,j] + m_Dz2[i,j]*CHz[i,j] + m_Dz4[i,j]*I_Dz[i,j]

        ## Ez Field Update
        for i in range(0,Nx,1):
            for j in range(0,Ny,1):
                Ez[i,j] = Pec[i,j]*(1/eps_zz[i,j])*Dz[i,j]

        # Ez Field Storage
        if t%10 == 0:
            for i in range(0,Nx,1):
                for j in range(0,Ny,1):
                    EzTime[int(t/10),i,j] = Ez[i,j]
            print(t)
        EzReciever[t] = np.sum(Ez[:,PML_Ly[0]+1])
    return
FDTD_Loop(EzTime,EzReciever,Ez,Dz,CEx,CEy,I_Dz,Hx,Hy,CHz,I_CEx,I_CEy)
timer_end = time.time()
print("Program took:", timer_end-timer_start-timer_delay[2], "seconds.")

# Timer logging below as required for analysis 
#with open("Samples/Timer.txt", "a") as f:
#    print("Execution on", time.strftime("%d %b %Y at %H:%M:%S took", time.localtime()), timer_end - timer_start, "seconds.", file=f)
#%%

# A potential data output method. This plots the magnitude of E at a constant position over time
plt.plot(t_sec,EzReciever)
#plt.plot(t_sec,EzTime[:,int(Nx/2)], label='Transmitted')
plt.title("|E| Over Time at Boundry")
plt.xlabel("Time")
plt.ylabel("|E|")
plt.savefig(fname="Ebound_Over_Time_TFSF_Source")
plt.show()

'''# Another data output method that calculates the frequencies transmitted and reflected by doing an FFT of Ey at boundries of grid and normalizing to the source
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
'''
#%%
np.save('EzTime_f.npy', EzTime)
ParamStore = np.array([X,Y,time_steps,Device_radius])
np.save('ParamStore_f.npy', ParamStore)

x = np.arange(0,Nx*dx,dx)

fps = 30
duration = time_steps/(fps*10*5)
fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(1,1,1)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)

print(EzTime.shape)
def make_frame(anim_time):

    time_step = anim_time*fps*5

    ax1.clear()
    
    #if (time_step/5 <= 20):
    #    im = ax1.imshow(EzTime[int(time_step),:,:], cmap='viridis')#, vmin=-0.01, vmax=0.01)
    #else: 
    im = ax1.imshow(EzTime[int(time_step),:,:], cmap='viridis', vmin=-1.5, vmax=1.5)
    rect = patches.Circle((550, 250), Device_radius, linewidth=1, edgecolor='w', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_title("Timestep = "+str(round(time_step*10)))
    fig.colorbar(im, cax=cax, orientation='vertical')

    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration = duration)

animation.ipython_display(fps = fps, loop = True, autoplay = True)
