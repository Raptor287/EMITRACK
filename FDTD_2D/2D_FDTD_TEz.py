#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

from mpl_toolkits.axes_grid1 import make_axes_locatable

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

#%%

# Timer to measure program Execution time
timer_start = time.time()
timer_delay = [0,0,0]                               # delay to compensate for user imput time. First 2 indexes are temp, 3rd stores total delay

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
n_max = np.sqrt(ur_max*er_max)                      # Max refractive index in grid
wavelength_min = c0/(f_max*n_max)                   # Min wavelength in grid
N_wave = 20                                         # Wave resolution
delta_wave = wavelength_min/N_wave                  # Grid resolution from wavelength
'''
## Structure Resolution
N_device = 1                                        # Device's smallest structure resolution
structure_min = 1                                   # Smallest structure is 1ft = 0.3048m
delta_device = structure_min/N_device               # Grid resolution from structure

## Initial Grid Resolution
dz = min(delta_wave,delta_device)                   # Initial res is smaller of wave/struct res

## Snapping Grid to Critical Dimension of Device
N_device = int(np.ceil(structure_min/dz))           # Number of cells to represent structure is now an integer
dz = structure_min/N_device                         # dz recalculated with N_device
'''
dx = dy = delta_wave                                     # Temporarily setting dz for tests without a device

# Building Grid

## Determining Grid Size
Nx = 200
Ny = 200

## Building Grid
xa = np.linspace(0,Nx-1,Nx)
ya = np.linspace(0,Ny-1,Ny)
X,Y = np.meshgrid(xa,ya)

## Building device
#Device_start = 2 + 500                              # 2 cells for source/record, 500 spacers
#Device_end = Device_start + N_device                # End of device indEy

### Mu
mu_xx = np.ones([Nx,Ny])
mu_yy = np.ones([Ny,Ny])
#mu_xx[Device_start:Device_end] = ur_max             # Array mu_xx contains permeablility across grid

### Eps
eps_zz = np.ones([Nx,Ny])
#eps_yy[Device_start:Device_end] = er_max            # Permittivity across grid

# Note: This is the device. Its simply represented by mu and eps values across the grid


# Time Params

dt = dx/(2*c0)                                      # dt such that the wave propogates one cell in 2 time steps (n_bounds*dz/(2*c0))
tprop = (n_max*Nx*dx)/c0                            # tprop is the approximate time for propogation across the grid


# Pulse Params
tau = 0.5/f_max                                     # 0.5/fmax, spread of pulse
t0 = 6*tau                                          # Time offset to ease into pulse


# Iteration Number Calculation
sim_time = 12*tau + 5*tprop                         # Total simulation time
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
#k_source = 1                                        # Location of pulse, array indEy friendly
#H_offset = ds/(2*c0) + 0.5*dt                       # (n_source*dz)/(2*c0) + (delta_t/2), 'Hx' source offset due to time/grid offset    
#H_scale = -np.sqrt(eps_yy[k_source]/mu_xx[k_source])    # -sqrt(e_rel/u_rel), normalization of 'Hx' source due to derivation of update Eqs
E_source = np.exp(-(((t_sec-t0)/tau)**2))               # Electric field source
#H_source = H_scale*np.exp(-(((t_sec-t0+H_offset)/tau)**2)) # Magnetic field source

# Update Coefficients
m_Hx = c0*dt/mu_xx
m_Hy = c0*dt/mu_yy
m_Dz = c0*dt
# Note: mu_xx and mu_yy are relative to u0


# Initializing Fields and Boundries

## Magnetic Fields
### The time arrays are very large (20Gb with 5000x5000 grid and 1000 steps). Use sparingly.
### 'dtype = np.half' uses half precision float to save memory (5gb with above params)
#HxTime = np.zeros((int(time_steps/10)+1,Nx,Ny),dtype=np.half); HyTime = np.zeros((int(time_steps/10+1),Nx,Ny),dtype=np.half); 
Hx = np.zeros((Nx,Ny)); Hy = np.zeros((Nx,Ny)); 
CHz = np.zeros((Nx,Ny))

## Electric Fields
EzTime = np.zeros((int(time_steps/10)+1,Nx,Ny))#,dtype=np.single)
Ez = np.zeros((Nx,Ny)); Dz = np.zeros((Nx,Ny))
CEx = np.zeros((Nx,Ny)); CEy = np.zeros((Nx,Ny))


# Main FDTD Loop
for t in range(0,time_steps+1,1):

    # Magnetic Field Update

    ## Ex and Ey Curl Updates
    for i in range(0,Nx,1):
        for j in range(0,Ny-1,1):
            CEx[i,j] = (Ez[i,j+1] - Ez[i,j])/dy
        CEx[i,Ny-1] = (0 - Ez[i,Ny-1])/dy
    
    for j in range(0,Ny,1):
        for i in range(0,Nx-1,1):
            CEy[i,j] = - (Ez[i+1,j] - Ez[i,j])/dx
        CEy[Nx-1,j] = - (0 - Ez[Nx-1,j])/dx

    ## Hx and Hy Updates
    for i in range(0,Nx,1):
        for j in range(0,Ny,1):
            Hx[i,j] = Hx[i,j] - m_Hx[i,j]*CEx[i,j]
            Hy[i,j] = Hy[i,j] - m_Hy[i,j]*CEy[i,j]
    
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

    ## Dz Field Update
    for i in range(0,Nx,1):
        for j in range(0,Ny,1):
            Dz[i,j] = Dz[i,j] + m_Dz*CHz[i,j]
    
    Dz[int(Nx/4),int(Ny/4)] = Dz[int(Nx/4),int(Ny/4)] + E_source[t]

    ## Ez Field Update
    for i in range(0,Nx,1):
        for j in range(0,Ny,1):
            Ez[i,j] = (1/eps_zz[i,j])*Dz[i,j]

    # Ez Field Storage
    if t%10 == 0:
        for i in range(0,Nx,1):
            for j in range(0,Ny,1):
                EzTime[int(t/10),i,j] = Ez[i,j]
        print(t)

timer_end = time.time()
print("Program took:", timer_end-timer_start-timer_delay[2], "seconds.")

# Timer logging below as required for analysis 
#with open("Samples/Timer.txt", "a") as f:
#    print("Execution on", time.strftime("%d %b %Y at %H:%M:%S took", time.localtime()), timer_end - timer_start, "seconds.", file=f)
#%%
'''
# A potential data output method. This plots the magnitude of E at a constant position over time
plt.plot(t_sec,Ey[:,0], label='Reflected')
plt.plot(t_sec,Ey[:,-1], label='Transmitted')
plt.legend()
plt.title("|E| Over Time at Boundries")
plt.xlabel("Time")
plt.ylabel("|E|")
plt.savefig(fname="Ebound_Over_Time_TFSF_Source")
plt.show()
'''
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

x = np.arange(0,Nx*dx,dx)

fps = 30
duration = time_steps/(fps*10)
fig, ax = plt.subplots() 
#ax = fig.add_subplot(111, projection='3d')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)

print(EzTime.shape)
def make_frame(anim_time):

    time_step = anim_time*fps

    ax.clear()
    #ax.plot(x, EzTime[int(time_step),:,49])
    im = ax.imshow(EzTime[int(time_step),:,:], cmap='jet')
    ax.set_title("Timestep = "+str(round(time_step*10)))
    fig.colorbar(im, cax=cax, orientation='vertical')

    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration = duration)

animation.ipython_display(fps = fps, loop = True, autoplay = True)
