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
N_wave = 20                                         # Wave resolution in cells
delta_wave = wavelength_min/N_wave                  # Grid resolution from wavelength
dx = dy = delta_wave                                
print("dx = dy =", delta_wave,"m")


# Building Grid

## Selecting Grid Size
### Nx2 and Ny2 are 2x grids that are used only to model the device. 'mu' and 'eps' will be pulled from 2x grid
Nx = 500; Nx2 = 2*Nx
Ny = 1000; Ny2 = 2*Ny

## Building Grid
xa = np.linspace(0,Nx-1,Nx)
ya = np.linspace(0,Ny-1,Ny)
X,Y = np.meshgrid(xa,ya)
### Lengths of the PML in the x and y directions
PML_Lx = np.array([20,20])
PML_Ly = np.array([20,20])

## Building Device
### X and Y position of cylinder in number of cells
Device_x = 250
Device_y = 550
### Cylinder radius in number of cells
Device_radius = int(np.ceil((c0/(2*np.pi*f0))/dx))           # Note the radius in meters is divided by dx to get the radius in cells rounded up
print("Device radius is:", Device_radius,"cells")
### Setting points inside cylinder radius to be perfect conductors
Pec = np.ones((Nx2,Ny2))
def Device(Device_radius):
    for i in range(0,Nx2,1):
        for j in range(0,Ny2,1):
            if ((i-Device_x)**2+(j-Device_y)**2 <= Device_radius**2):
                Pec[:,j] = 0
    return
Device(Device_radius)

### Mu
mu_zz = np.ones((Nx,Ny))

### Eps
eps_xx = np.ones((Nx,Ny))
eps_yy = np.ones((Nx,Ny))


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
H_scale = -np.sqrt(eps_yy[int(Nx/2),j_source]/mu_zz[int(Nx/2),j_source])    # -sqrt(e_rel/u_rel), normalization of 'Hx' source due to derivation of update Eqs
## Pulse shaping using gaussian to ease in/out of source
Ex_source_shape = np.exp(-(((t_sec-t0)/tau)**2))
Hz_source_shape = np.exp(-(((t_sec-t0+H_offset)/tau)**2))
for t in range(0,time_steps+1,1):
    if t*dt > t0 and t*dt < t0+pulse_width:
        Ex_source_shape[t] = 1
        Hz_source_shape[t] = 1
    elif t*dt >= t0+pulse_width:
        Ex_source_shape[t] = np.exp(-(((t_sec[t]-t0-pulse_width)/tau)**2))
        Hz_source_shape[t] = np.exp(-(((t_sec[t]-t0-pulse_width+H_offset)/tau)**2))

## Source functions
Ex_source = Ex_source_shape*np.cos(2*np.pi*f0*t_sec)          # Electric field source
Hz_source = Hz_source_shape*H_scale*np.cos(2*np.pi*f0*(t_sec+H_offset)) # Magnetic field source


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

## Hz Coefficients
sigHx = sig_x[1::2,1::2]; sigHy = sig_y[1::2,1::2]
m_Hz0 = (1/dt) + ((sigHx + sigHy)/(2*e0)) + (sigHx*sigHy*dt/(4*(e0**2)))
m_Hz1 = (1/dt) - ((sigHx + sigHy)/(2*e0)) - (sigHx*sigHy*dt/(4*(e0**2))); m_Hz1 = m_Hz1/m_Hz0
m_Hz2 = - c0/(mu_zz*m_Hz0)
m_Hz4 = - (dt/(e0**2))*sigHx*sigHy/m_Hz0

## Dx Coefficients
sigDx = sig_x[1::2,0::2]; sigDy = sig_y[1::2,0::2]
m_Dx0 = (1/dt) + sigDy/(2*e0)
m_Dx1 = ((1/dt) - sigDy/(2*e0))/m_Dx0
m_Dx2 = c0/(m_Dx0)
m_Dx3 = (c0*dt/e0) * sigDx/(m_Dx0)

## Dy Coefficients
sigDx = sig_x[0::2,1::2]; sigDy = sig_y[0::2,1::2]
m_Dy0 = (1/dt) + sigDx/(2*e0)
m_Dy1 = ((1/dt) - sigDx/(2*e0))/m_Dy0
m_Dy2 = c0/(m_Dy0)
m_Dy3 = (c0*dt/e0) * sigDy/(m_Dy0)


# Initializing Fields and Boundries

## Magnetic Fields
### The time arrays are very large (20Gb with 5000x5000 grid and 1000 steps). Use sparingly.
### 'dtype = np.half' uses half precision float to save memory (5gb with above params)
#HxTime = np.zeros((int(time_steps/10)+1,Nx,Ny),dtype=np.half); HyTime = np.zeros((int(time_steps/10+1),Nx,Ny),dtype=np.half); 
Hz = np.zeros((Nx,Ny))
I_Hz = np.zeros((Nx,Ny))
CHx = np.zeros((Nx,Ny)); CHy = np.zeros((Nx,Ny))

## Electric Fields
ExTime = np.zeros((int(time_steps/10)+1,Nx,Ny),dtype=np.single)
#EyTime = np.zeros((int(time_steps/10)+1,Nx,Ny),dtype=np.single)
ExReciever = np.zeros(time_steps+1)
Ex = np.zeros((Nx,Ny)); Dx = np.zeros((Nx,Ny))
Ey = np.zeros((Nx,Ny)); Dy = np.zeros((Nx,Ny))
I_CHx = np.zeros((Nx,Ny)); I_CHy = np.zeros((Nx,Ny))
CEz = np.zeros((Nx,Ny))


# Main FDTD Loop
@jit(nopython=True)
def FDTD_Loop(ExTime,ExReciever,Ex,Ey,Dx,Dy,CEz,Hz,CHx,CHy,I_CHx,I_CHy,I_Hz):
    for t in range(0,time_steps+1,1):

        # Magnetic Field Update

        ## Ez Curl Update
        ### (Nx,Ny) Corner Update
        CEz[Nx-1,Ny-1] = (0 - Ey[Nx-1,Ny-1])/dx - (0 - Ex[Nx-1,Ny-1])/dy
        ### y=Ny Row Update
        for i in range(0,Nx-1,1):
            CEz[i,Ny-1] = (Ey[i+1,Ny-1] - Ey[i,Ny-1])/dx - (0 - Ex[i,Ny-1])/dy
        ### Remaining Grid Update
        for j in range(0,Ny-1,1):
            CEz[Nx-1,j] = (0 - Ey[Nx-1,j])/dx - (Ex[Nx-1,j+1] - Ex[Nx-1,j])/dy
            for i in range(0,Nx-1,1):
                CEz[i,j] = (Ey[i+1,j] - Ey[i,j])/dx - (Ex[i,j+1] - Ex[i,j])/dy

        for i in range(0,Nx,1):
            CEz[i,j_source-1] = CEz[i,j_source-1] + Ex_source[t]/dy

        ## Hz Integration
        I_Hz = I_Hz + Hz

        ## Hz Field Update
        for i in range(0,Nx,1):
            for j in range(0,Ny,1):
                Hz[i,j] = m_Hz1[i,j]*Hz[i,j] + m_Hz2[i,j]*CEz[i,j] + m_Hz4[i,j]*I_Hz[i,j]
        

        # Electric Field Update

        ## Hx and Hy Curl Updates
        for i in range(0,Nx,1):
            for j in range(1,Ny,1):
                CHx[i,j] = (Hz[i,j] - Hz[i,j-1])/dy
            CHx[i,0] = (Hz[i,0] - 0)/dy
        
        for i in range(0,Nx,1):
            CHx[i,j_source] = CHx[i,j_source] - Hz_source[t]/dy
        
        for j in range(0,Ny,1):
            for i in range(1,Nx,1):
                CHy[i,j] = - (Hz[i,j] - Hz[i-1,j])/dx
            CHy[0,j] = - (Hz[0,j] - 0)/dx
        
        ## CHx and CHy Integrations
        I_CHx = I_CHx + CHx
        I_CHy = I_CHy + CHy

        ## Dx and Dy Updates
        for i in range(0,Nx,1):
            for j in range(0,Ny,1):
                Dx[i,j] = m_Dx1[i,j]*Dx[i,j] + m_Dx2[i,j]*CHx[i,j] + m_Dx3[i,j]*I_CHx[i,j]
                Dy[i,j] = m_Dy1[i,j]*Dy[i,j] + m_Dy2[i,j]*CHy[i,j] + m_Dy3[i,j]*I_CHy[i,j]

        ## Ex and Ey Field Update
        for i in range(0,Nx,1):
            for j in range(0,Ny,1):
                Ex[i,j] = Pec[i,j]*(1/eps_xx[i,j])*Dx[i,j]
                Ey[i,j] = Pec[i,j]*(1/eps_yy[i,j])*Dy[i,j]

        # Ex Field Storage
        if t%10 == 0:
            for i in range(0,Nx,1):
                for j in range(0,Ny,1):
                    ExTime[int(t/10),i,j] = Ex[i,j]
            print(t)
        ExReciever[t] = np.sum(Ex[(PML_Lx[0]+2):(Nx-PML_Lx[1]-2),PML_Ly[0]+2])
    return
FDTD_Loop(ExTime,ExReciever,Ex,Ey,Dx,Dy,CEz,Hz,CHx,CHy,I_CHx,I_CHy,I_Hz)
timer_end = time.time()
print("Program took:", timer_end-timer_start-timer_delay[2], "seconds.")

# Timer logging below as required for analysis 
#with open("Samples/Timer.txt", "a") as f:
#    print("Execution on", time.strftime("%d %b %Y at %H:%M:%S took", time.localtime()), timer_end - timer_start, "seconds.", file=f)
#%%

# This plots the magnitude of E at a constant position over time
plt.plot(t_sec,ExReciever)
plt.title("|E| Recieved Over Time")
plt.annotate("Total Energy Recieved: "+str(round(np.sum(np.abs(ExReciever[:])),3)), xy=(t_sec[int(int(t_sec.shape[0])/2)],ExReciever.max()))
plt.xlabel("Time")
plt.ylabel("|E|")
plt.savefig(fname="Ebound_Over_Time_TFSF_Source")
plt.show()

#%%
# Saving the EzTime array and some parameters for use in ExTimeAnimation.py
np.save('EzTime_f.npy', ExTime)
ParamStore = np.array([X,Y,time_steps,Device_radius])
np.save('ParamStore_f.npy', ParamStore)

# Short preview animation
fps = 30
duration = time_steps/(fps*10*5)
fig = plt.figure(figsize=(16,9))
ax1 = fig.add_subplot(1,1,1)
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)

print(ExTime.shape)
def make_frame(anim_time):

    time_step = anim_time*fps*5

    ax1.clear()
    
    #if (time_step/5 <= 20):
    #    im = ax1.imshow(EzTime[int(time_step),:,:], cmap='viridis')#, vmin=-0.01, vmax=0.01)
    #else: 
    im = ax1.imshow(ExTime[int(time_step),:,:], cmap='viridis', vmin=-1.5, vmax=1.5)
    rect = patches.Circle((550, 250), Device_radius, linewidth=1, edgecolor='w', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_title("Timestep = "+str(round(time_step*10)))
    fig.colorbar(im, cax=cax, orientation='vertical')

    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration = duration)

animation.ipython_display(fps = fps, loop = True, autoplay = True)
