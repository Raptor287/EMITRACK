# EMITRACK
UAF Physics Capstone Project EMITRACK - Electromagnetic Wave-Meteor Interaction Tracking, Analysis, and Recording Kit

EMITRACK is a finite-difference time-domain program designed to model radar waves interacting with overdense meteor trails.
Documentation and project details may be found in `Capstone_Report.pdf`


## Organization

### FDTD_1D

This folder contains an FDTD program with an object of infinite length in two dimensions. The wave propogates along the z-axis. The finalized working program can be found at `Sample_Problem/1D_FDTD_Sample_Problem.py`. Some sample outputs can be found in the corresponding `Samples` folder.

### FDTD_2D

The 2D folder contains the finalized model. The model is split into two folders: `2D_TE_PML` containing the simulation for electric fields polarized in the x-direction, and `2D_TM_PML` containing a simulation of electric fields polarized in the z-direction.

### Python Testing

This folder contains code to build geometries in the 2D FDTD simulations.

## Usage

### Python 3.10

* This program requires Python 3.10 which can be downloaded at https://www.python.org/downloads/
* After downloading and installing Python 3.10, create a virtual environment from a command terminal using `python -m venv /path/to/new/virtual/environment`. 
    * This will allow you to isolate the required libraries on your computer.
* Next, activate the virtual environment. On Windows Powershell, this can be done using the command `PS C:\> <venv>\Scripts\Activate.ps1`. 
* Finally, install the required libraries using:
```
python -m pip install matplotlib
python -m pip install moviepy
python -m pip install numba
```
With that, Python should be configured for use with EMITRACK. Note that the virtual environment must be re-activated every time the terminal is restarted.


### EMITRACK

* To use EMITRACK programs, first open the desired python file using a code editor like "VS Code" or "Notepad ++". 
* Next, modify the program using the documentation in `Capstone_Report.pdf` and the comments in the code as a reference to fit the desired parameters.
* Once the file is configured, save and exit.
* Run the file from the command line using `python FILE_YOU_WANT_TO_RUN.py`