# Python header (please make sure you are working with Python 3)
import numpy as np # numerical python
from numpy import pi as PI
import pylab as plt # replica of matlab's plotting tools
import scipy
import sys
from sys import getsizeof # accesses system tools
from pdb import set_trace # this stops the code and gives you interactive env.
from scipy.interpolate import pchip_interpolate
import matplotlib.animation as animation

# LaTeX setup
from matplotlib import rc as matplotlibrc
from matplotlib import patches
matplotlibrc('text',usetex=True)
matplotlibrc('font', family='serif')

# compressible fluid model
Rgas = 287.0 # J/kg/K
gamma = 1.4
Cv = Rgas/(gamma-1.0) # J/kg/K
Cp = Cv*gamma # assume calorically perfect

P0   = 101000 # Pa, base quiescent air pressure
rho0  = 1.2 # kg/m^3
T0 = P0/rho0/Rgas; # Kelvin
a0 = np.sqrt(gamma*Rgas*T0) # base speed of sound, not the instantaneous/true speed

loc_xrel = 30
loc_yrel = 35

# Grid generation
Lx  = 2  # characteristic length scale
Ly  = 1
Nxc = 100 # number of cells
Nyc = 70

Nxf = Nxc+1 # number of faces = number of cells plus 1
Nyf = Nyc+1 # number of faces = number of cells plus 1
xf = np.linspace(0,Lx,Nxf) # linearly spaced grid for faces
yf = np.linspace(0,Ly,Nyf) # linearly spaced grid for faces
dx = xf[1]-xf[0]
dy = yf[1]-yf[0]

# cell-centers array in x
xc_int = 0.5*(xf[:-1]+xf[1:]) # internal cell center locations
Ng = 2 # number of ghost cells
if Ng > Nxc:
    sys.exit("Too many ghost cells!")
if Ng == 1:
   sys.exit("Need to run with at least 2 ghost cells!")
xc_ghosts_left = xc_int[:Ng]-Ng*dx
xc_ghosts_right = xc_int[-Ng:]+Ng*dx
xc = np.append(xc_ghosts_left,xc_int)
xc = np.append(xc,xc_ghosts_right) # final xc array

# cell-centers array in x
yc_int = 0.5*(yf[:-1]+yf[1:]) # internal cell center locations
if Ng > Nyc:
    sys.exit("Too many ghost cells!")
yc_ghosts_left = yc_int[:Ng]-Ng*dy
yc_ghosts_right = yc_int[-Ng:]+Ng*dy
yc = np.append(yc_ghosts_left,yc_int)
yc = np.append(yc,yc_ghosts_right) # final yc array

Xc,Yc=np.meshgrid(xc,yc) # this create two, 2D arrays with coordinate values
# these are indexed as: Xc[j,i],Yc[j,i]
# Xf,Yf=np.meshgrid(xf,yf) # this create two, 2D arrays with coordinate values

# flux_O1 = 0.0*np.array(xf) # flux array is indexed from 0-->Nxf

Frho_x  = np.zeros([Nyc,Nxf])
Frhou_x = np.zeros([Nyc,Nxf])
Frhov_x = np.zeros([Nyc,Nxf])
FE_x    = np.zeros([Nyc,Nxf])

Frho_y  = np.zeros([Nyf,Nxc])
Frhou_y = np.zeros([Nyf,Nxc])
Frhov_y = np.zeros([Nyf,Nxc])
FE_y    = np.zeros([Nyf,Nxc])

Frho_x_star  = np.zeros([Nyc,Nxf]) # Nxf = Nxc+1
Frhou_x_star = np.zeros([Nyc,Nxf])
Frhov_x_star = np.zeros([Nyc,Nxf])
FE_x_star    = np.zeros([Nyc,Nxf])

Frho_y_star  = np.zeros([Nyf,Nxc])
Frhou_y_star = np.zeros([Nyf,Nxc])
Frhov_y_star = np.zeros([Nyf,Nxc])
FE_y_star    = np.zeros([Nyf,Nxc])

# define initial conditions
lambda_x,lambda_y = Lx*2.0,Ly*2.0

f_xy = np.cos(2*PI/lambda_x*Xc)*np.cos(2*PI/lambda_y*Yc) # shape of the initial conditions
Amp = 0 # linear acoustic, gentle sound, linear advection
p_fluctuation = Amp*rho0*a0*a0*f_xy # Pa, p'(y,x,t=0)
p_init =  p_fluctuation + P0 # Pa, total initial pressure
rho_init = p_fluctuation/a0/a0 + rho0 # kg/m^3
T_init = p_init/rho_init/Rgas; # applying equation of state
u_init = 0.0*Xc # Amp*a0*f_x # m/s
v_init = 0.0*Yc # Amp*a0*f_x # m/s
sie = Cv*T_init # specific internal energy, J/kg
energy_init = rho_init*sie + 0.5*rho_init*(u_init*u_init+v_init*v_init) # J/m^3

Q_init = np.stack([rho_init,
               rho_init*u_init,
               rho_init*v_init,
               energy_init])

# # dimensions are: Q[var_index, j-y , i-x], and it includes ghost/guard cells

# update initial velocity for inlet and outlet
u_des = 5
u_in = u_des
u_out = u_des
u_init[:,:Ng] = u_in
u_init[:,-Ng:] = u_out

base = 4
y_comp = base + 3
y_comp2 = y_comp - 1
y_comp1 = y_comp - 2

# working variables
rho = np.array(rho_init)
u = np.array(u_init)
v = np.array(v_init)
energy = np.array(energy_init)
Qvec = np.array(Q_init)
rhou = np.array(Qvec[1,:,:])
rhov = np.array(Qvec[2,:,:])
p = np.array(p_init)

rho_star = np.array(rho_init)
u_star = np.array(u_init)
v_star = np.array(v_init)
energy_star = np.array(energy_init)
Qvec_star = np.array(Q_init)
rhou_star = np.array(Qvec_star[1,:,:])
rhov_star = np.array(Qvec_star[2,:,:])
p_star = np.array(p_init)


# Sponge Layer
sponge_cells = 15 # cell width
sigma = (np.linspace(0,1,sponge_cells))**2

def sponge_layer(rho,rhou,rhov,energy):
  rho[:, -sponge_cells:] = rho[:, -sponge_cells:] * (1-sigma) + \
                rho_init[:, -sponge_cells:] * sigma
  rhou[:, -sponge_cells:] = rhou[:, -sponge_cells:] * (1-sigma) + \
                Q_init[1,:, -sponge_cells:] * sigma
  rhov[:, -sponge_cells:] = rhov[:, -sponge_cells:] * (1-sigma) + \
                Q_init[2,:, -sponge_cells:] * sigma
  energy[:, -sponge_cells:] = energy[:, -sponge_cells:] * (1-sigma) + \
                energy_init[:, -sponge_cells:] * sigma
  return rho,rhou,rhov,energy


def update_ghost_cells(rho,rhou,rhov,p,u,energy,v):
    
    ############################ Obstacle Walls NaNs ############################
    # Block ghost cells -- horizontal
    rho[loc_yrel-y_comp1:loc_yrel+y_comp2,loc_xrel] = np.nan
    rhou[loc_yrel-y_comp1:loc_yrel+y_comp2,loc_xrel] = np.nan
    rhov[loc_yrel-y_comp1:loc_yrel+y_comp2,loc_xrel] = np.nan
    p[loc_yrel-y_comp1:loc_yrel+y_comp2,loc_xrel] = np.nan
    u[loc_yrel-y_comp1:loc_yrel+y_comp2,loc_xrel] = np.nan
    energy[loc_yrel-y_comp1:loc_yrel+y_comp2,loc_xrel] = np.nan
    v[loc_yrel-y_comp1:loc_yrel+y_comp2,loc_xrel] = np.nan
    ############################ Boundary Walls NaNs ############################
    
    
    # NaN cell filling -- West Side\
    u[:,:Ng] = u_in
    rho[:,:Ng] = 1.5*rho[:,Ng-1:Ng+1]-0.5*rho[:,Ng:Ng+2]
    rhou[:,:Ng] = rho[:,:Ng]*u[:,:Ng]
    v[:,:Ng] = 0.0
    rhov[:,:Ng] = rho[:,:Ng]*v[:,:Ng]
    p[:,:Ng] = 1.5*p[:,Ng-1:Ng+1]-0.5*p[:,Ng:Ng+2]
    energy[:,:Ng] = rho[:,:Ng]*Cv*(p[:,:Ng]/rho[:,:Ng]/Rgas) + \
                    0.5*rho[:,:Ng]*(u[:,:Ng]**2 + v[:,:Ng]**2)
  
    # NaN cell filling -- South Side
    u[:Ng,:] = 0.0
    v[:Ng,:] = 0.0
    rho[:Ng,:] = 1.5*rho[Ng-1:Ng+1,:]-0.5*rho[Ng:Ng+2,:]
    rhou[:Ng,:] = rho[:Ng,:]*u[:Ng,:]
    rhov[:Ng,:] = rho[:Ng,:]*v[:Ng,:]
    p[:Ng,:] = 1.5*p[Ng-1:Ng+1,:]-0.5*p[Ng:Ng+2,:]
    energy[:Ng,:] = rho[:Ng,:]*Cv*(p[:Ng,:]/rho[:Ng,:]/Rgas) + \
                    0.5*rho[:Ng,:]*(u[:Ng,:]**2 + v[:Ng,:]**2)
        
    # NaN cell filling -- North Side
    rho[-Ng:,:] = 1.5*rho[-Ng-2:-2,:]-0.5*rho[-Ng-1:-1,:]
    u[-Ng:,:] = 0.0
    v[-Ng:,:] = 0.0
    rhou[-Ng:,:] = rho[-Ng:,:]*u[-Ng:,:]  
    rhov[-Ng:,:] = rho[-Ng:,:]*v[-Ng:,:]
    p[-Ng:,:] = 1.5*p[-Ng-2:-2,:]-0.5*p[-Ng-1:-1,:]
    energy[-Ng:,:] = rho[-Ng:,:]*Cv*(p[-Ng:,:]/rho[-Ng:,:]/Rgas) + \
                    0.5*rho[-Ng:,:]*(u[-Ng:,:]**2 + v[-Ng:,:]**2)

    # NaN cell filling -- East Side
    u[:,-Ng:] = u_out
    rho[:,-Ng:] = 1.5*rho[:,-Ng-2:-2]-0.5*rho[:,-Ng-1:-1]
    rhou[:,-Ng:] = 1.5*rhou[:,-Ng-2:-2]-0.5*rhou[:,-Ng-1:-1]
    v[:,-Ng:] = 1.5*v[:,-Ng-2:-2]-0.5*v[:,-Ng-1:-1]
    rhov[:,-Ng:] = 1.5*rhov[:,-Ng-2:-2]-0.5*rhov[:,-Ng-1:-1]
    p[:,-Ng:] = 1.5*p[:,-Ng-2:-2]-0.5*p[:,-Ng-1:-1]
    energy[:,-Ng:] = rho[:,-Ng:]*Cv*(p[:,-Ng:]/rho[:,-Ng:]/Rgas) + \
                    0.5*rho[:,-Ng:]*(u[:,-Ng:]**2 + v[:,-Ng:]**2)

    return rho, rhou, rhov, p, u, energy, v
    

def block_flux(Frho_x,Frhou_x,Frhov_x,FE_x,Frho_y,Frhou_y,Frhov_y,FE_y):
    # South
    Frho_y[loc_yrel+base,loc_xrel-2] = 0.0
    Frhou_y[loc_yrel+base,loc_xrel-2] = 0.0
    Frhov_y[loc_yrel+base,loc_xrel-2] = 1.5*p[loc_yrel+y_comp2,loc_xrel] - 0.5*p[loc_yrel+y_comp2+1,loc_xrel]
    FE_y[loc_yrel+base,loc_xrel-2]= 0.0

    # North
    Frho_y[loc_yrel-y_comp,loc_xrel-2] = 0.0
    Frhou_y[loc_yrel-y_comp,loc_xrel-2] = 0.0
    Frhov_y[loc_yrel-y_comp,loc_xrel-2] = 1.5*p[loc_yrel-y_comp2,loc_xrel] - 0.5*p[loc_yrel-y_comp2-1,loc_xrel]
    FE_y[loc_yrel-y_comp,loc_xrel-2] = 0.0

    # East
    Frho_x[loc_yrel-y_comp:loc_yrel+base,loc_xrel-2] = 0.0
    Frhou_x[loc_yrel-y_comp:loc_yrel+base,loc_xrel-2] = 1.5*p[loc_yrel-y_comp1:loc_yrel+y_comp2,loc_xrel-1] - 0.5*p[loc_yrel-y_comp1:loc_yrel+y_comp2,loc_xrel-2]
    Frhov_x[loc_yrel-y_comp:loc_yrel+base,loc_xrel-2] = 0.0
    FE_x[loc_yrel-y_comp:loc_yrel+base,loc_xrel-2] = 0.0

    # West
    Frho_x[loc_yrel-y_comp:loc_yrel+base,loc_xrel-1] = 0.0
    Frhou_x[loc_yrel-y_comp:loc_yrel+base,loc_xrel-1] = 1.5*p[loc_yrel-y_comp1:loc_yrel+y_comp2,loc_xrel+1] - 0.5*p[loc_yrel-y_comp1:loc_yrel+y_comp2,loc_xrel+2]
    Frhov_x[loc_yrel-y_comp:loc_yrel+base,loc_xrel-1] = 0.0
    FE_x[loc_yrel-y_comp:loc_yrel+base,loc_xrel-1] = 0.0

    return Frho_x,Frhou_x,Frhov_x,FE_x,Frho_y,Frhou_y,Frhov_y,FE_y


Nt = 8000
fig_skip = 100
CFL= 0.4
dt = CFL*dx/a0

figure_counter = 0

# note on indexing:


for it in range(0,Nt):
    
  ## populating ghost cells based on boundary conditions
  rho,rhou,rhov,energy = sponge_layer(rho,rhou,rhov,energy)

  rho,rhou,rhov,p,u,energy,v = update_ghost_cells(rho,rhou,rhov,p,u,energy,v)

  print("RK2 intervention: Obtaining F^n --> Q_star ")
    
  # RK2 intervention: Obtaining F^n --> Q_star
  Frho_x = 0.5 * (rhou[Ng:-Ng, Ng-1:-Ng] + rhou[Ng:-Ng, Ng:-Ng+1]) \
          - 0.25 * dx/dt * (rho[Ng:-Ng, Ng:-Ng+1] - rho[Ng:-Ng, Ng-1:-Ng])  # flux of density
  
  Frho_y = 0.5 * (rhov[Ng-1:-Ng, Ng:-Ng] + rhov[Ng:-Ng+1, Ng:-Ng]) \
          - 0.25 * dy/dt * (rho[Ng:-Ng+1, Ng:-Ng] - rho[Ng-1:-Ng, Ng:-Ng])  # flux of density

  # Frho_x[:, 0], Frho_x[:, -1] = 0.0, 0.0
  Frho_y[0, :], Frho_y[-1, :] = 0.0, 0.0
  
  # u-momentum
  Frhou_x = Frho_x * 0.5 * (u[Ng:-Ng, Ng-1:-Ng] + u[Ng:-Ng, Ng:-Ng+1]) \
          + 0.5 * (p[Ng:-Ng, Ng-1:-Ng] + p[Ng:-Ng, Ng:-Ng+1]) \
          - 0.25 * dx/dt * (rhou[Ng:-Ng, Ng:-Ng+1] - rhou[Ng:-Ng, Ng-1:-Ng])
  # Frhou_x[:, 0] = 1.5 * p[Ng:-Ng, Ng] - 0.5 * p[Ng:-Ng, Ng+1]
  # Frhou_x[:, -1] = 1.5 * p[Ng:-Ng, -Ng-1] - 0.5 * p[Ng:-Ng, -Ng-2]

  Frhou_y = Frho_y * 0.5 * (u[Ng-1:-Ng, Ng:-Ng] + u[Ng:-Ng+1, Ng:-Ng]) \
            - 0.25 * dy/dt * (rhou[Ng:-Ng+1, Ng:-Ng] - rhou[Ng-1:-Ng, Ng:-Ng])
  Frhou_y[0, :], Frhou_y[-1, :] = 0.0, 0.0

  # v-momentum
  Frhov_x = Frho_x * 0.5 * (v[Ng:-Ng, Ng-1:-Ng] + v[Ng:-Ng, Ng:-Ng+1]) \
          - 0.25 * dx/dt * (rhov[Ng:-Ng, Ng:-Ng+1] - rhov[Ng:-Ng, Ng-1:-Ng])
  # Frhov_x[:, 0], Frhov_x[:, -1] = 0.0, 0.0
  Frhov_x[:, 0] = 0.0


  Frhov_y = Frho_y * 0.5 * (v[Ng-1:-Ng, Ng:-Ng] + v[Ng:-Ng+1, Ng:-Ng]) \
          + 0.5 * (p[Ng-1:-Ng, Ng:-Ng] + p[Ng:-Ng+1, Ng:-Ng]) \
          - 0.25 * dy/dt * (rhov[Ng:-Ng+1, Ng:-Ng] - rhov[Ng-1:-Ng, Ng:-Ng])
  Frhov_y[0, :] = 1.5 * p[Ng, Ng:-Ng] - 0.5 * p[Ng+1, Ng:-Ng]  # linear extrapolation to inform momentum flux condition at the wall
  Frhov_y[-1, :] = 1.5 * p[-Ng-1, Ng:-Ng] - 0.5 * p[-Ng-2, Ng:-Ng]

  # p[Ng,-Ng-1]
  # energy
  tmpx, tmpy = u * (energy + p), v * (energy + p)
  FE_x = 0.5 * (tmpx[Ng:-Ng, Ng-1:-Ng] + tmpx[Ng:-Ng, Ng:-Ng+1]) \
          - 0.25 * dx/dt * (energy[Ng:-Ng, Ng:-Ng+1] - energy[Ng:-Ng, Ng-1:-Ng])
  FE_y = 0.5 * (tmpy[Ng-1:-Ng, Ng:-Ng] + tmpy[Ng:-Ng+1, Ng:-Ng]) \
          - 0.25 * dy/dt * (energy[Ng:-Ng+1, Ng:-Ng] - energy[Ng-1:-Ng, Ng:-Ng])

  # boundary conditions (modifying flux and ghost cells)
  # FE_x[:, 0], FE_x[:, -1] = 0.0, 0.0
  FE_y[0, :], FE_y[-1, :] = 0.0, 0.0


  # Introduce block fluxes
  Frho_x,Frhou_x,Frhov_x,FE_x,Frho_y,Frhou_y,Frhov_y,FE_y = block_flux(Frho_x,Frhou_x,Frhov_x,FE_x,Frho_y,Frhou_y,Frhov_y,FE_y)
  
#   set_trace()
  # RK2 intervention: Obtaining starred rho, rhou, rhov, energy
  rho_star[Ng:-Ng, Ng:-Ng] = rho[Ng:-Ng, Ng:-Ng] \
          - dt * (Frho_x[:, 1:] - Frho_x[:, :-1]) / dx \
          - dt * (Frho_y[1:, :] - Frho_y[:-1, :]) / dy

  rhou_star[Ng:-Ng, Ng:-Ng] = rhou[Ng:-Ng, Ng:-Ng] \
              - dt * (Frhou_x[:, 1:] - Frhou_x[:, :-1]) / dx \
              - dt * (Frhou_y[1:, :] - Frhou_y[:-1, :]) / dy

  rhov_star[Ng:-Ng, Ng:-Ng] = rhov[Ng:-Ng, Ng:-Ng] \
              - dt * (Frhov_x[:, 1:] - Frhov_x[:, :-1]) / dx \
              - dt * (Frhov_y[1:, :] - Frhov_y[:-1, :]) / dy

  energy_star[Ng:-Ng, Ng:-Ng] = energy[Ng:-Ng, Ng:-Ng] \
              - dt * (FE_x[:, 1:] - FE_x[:, :-1]) / dx \
              - dt * (FE_y[1:, :] - FE_y[:-1, :]) / dy

  # RK2 intervention: Obtaining starred u, v, kinetic_energy, p
  u_star = rhou_star / rho_star
  v_star = rhov_star / rho_star
  kinetic_energy_star = 0.5 * (rhou_star * rhou_star + rhov_star * rhov_star) / rho_star
  p_star = (energy_star - kinetic_energy_star) * (gamma - 1.0)


  # RK2 intervention: update ghost cells for all quantities
  rho_star,rhou_star,rhov_star,energy_star = sponge_layer(rho_star,rhou_star,rhov_star,energy_star)
  rho_star,rhou_star,rhov_star,p_star,u_star,energy_star,v_star = update_ghost_cells(rho_star,rhou_star,rhov_star,p_star,u_star,energy_star,v_star)

  # RK2 intervention: update variables
  # 222222 RK2 ad construct starred Flux......
  Frho_x_star = 0.5 * (rhou_star[Ng:-Ng, Ng-1:-Ng] + rhou_star[Ng:-Ng, Ng:-Ng+1]) \
          - 0.25 * dx/dt * (rho_star[Ng:-Ng, Ng:-Ng+1] - rho_star[Ng:-Ng, Ng-1:-Ng])  # flux of density

  Frho_y_star = 0.5 * (rhov_star[Ng-1:-Ng, Ng:-Ng] + rhov_star[Ng:-Ng+1, Ng:-Ng]) \
          - 0.25 * dy/dt * (rho_star[Ng:-Ng+1, Ng:-Ng] - rho_star[Ng-1:-Ng, Ng:-Ng])  # flux of density

  # Frho_x_star[:, 0], Frho_x_star[:, -1] = 0.0, 0.0
  Frho_y_star[0, :], Frho_y_star[-1, :] = 0.0, 0.0

  # u-momentum
  Frhou_x_star = Frho_x_star * 0.5 * (u_star[Ng:-Ng, Ng-1:-Ng] + u_star[Ng:-Ng, Ng:-Ng+1]) \
          + 0.5 * (p_star[Ng:-Ng, Ng-1:-Ng] + p_star[Ng:-Ng, Ng:-Ng+1]) \
          - 0.25 * dx/dt * (rhou_star[Ng:-Ng, Ng:-Ng+1] - rhou_star[Ng:-Ng, Ng-1:-Ng])
  # Frhou_x_star[:, 0] = 1.5 * p_star[Ng:-Ng, Ng] - 0.5 * p_star[Ng:-Ng, Ng+1]
  # Frhou_x_star[:, -1] = 1.5 * p_star[Ng:-Ng, -Ng-1] - 0.5 * p_star[Ng:-Ng, -Ng-2]

  Frhou_y_star = Frho_y_star * 0.5 * (u_star[Ng-1:-Ng, Ng:-Ng] + u_star[Ng:-Ng+1, Ng:-Ng]) \
          - 0.25 * dy/dt * (rhou_star[Ng:-Ng+1, Ng:-Ng] - rhou_star[Ng-1:-Ng, Ng:-Ng])
  Frhou_y_star[0, :], Frhou_y_star[-1, :] = 0.0, 0.0

  # v-momentum
  Frhov_x_star = Frho_x_star * 0.5 * (v_star[Ng:-Ng, Ng-1:-Ng] + v_star[Ng:-Ng, Ng:-Ng+1]) \
          - 0.25 * dx/dt * (rhov_star[Ng:-Ng, Ng:-Ng+1] - rhov_star[Ng:-Ng, Ng-1:-Ng])
  # Frhov_x_star[:, 0], Frhov_x_star[:, -1] = 0.0, 0.0
  Frhov_x_star[:, 0] = 0.0


  Frhov_y_star = Frho_y_star * 0.5 * (v_star[Ng-1:-Ng, Ng:-Ng] + v_star[Ng:-Ng+1, Ng:-Ng]) \
          + 0.5 * (p_star[Ng-1:-Ng, Ng:-Ng] + p_star[Ng:-Ng+1, Ng:-Ng]) \
          - 0.25 * dy/dt * (rhov_star[Ng:-Ng+1, Ng:-Ng] - rhov_star[Ng-1:-Ng, Ng:-Ng])
  Frhov_y_star[0, :] = 1.5 * p_star[Ng, Ng:-Ng] - 0.5 * p_star[Ng+1, Ng:-Ng]  # linear extrapolation to inform momentum flux condition at the wall
  Frhov_y_star[-1, :] = 1.5 * p_star[-Ng-1, Ng:-Ng] - 0.5 * p_star[-Ng-2, Ng:-Ng]

  tmpx_star, tmpy_star = u_star * (energy_star + p_star), v_star * (energy_star + p_star)
  FE_x_star = 0.5 * (tmpx_star[Ng:-Ng, Ng-1:-Ng] + tmpx_star[Ng:-Ng, Ng:-Ng+1]) \
          - 0.25 * dx/dt * (energy_star[Ng:-Ng, Ng:-Ng+1] - energy_star[Ng:-Ng, Ng-1:-Ng])  
  FE_y_star = 0.5 * (tmpy_star[Ng-1:-Ng, Ng:-Ng] + tmpy_star[Ng:-Ng+1, Ng:-Ng]) \
          - 0.25 * dy/dt * (energy_star[Ng:-Ng+1, Ng:-Ng] - energy_star[Ng-1:-Ng, Ng:-Ng])
  
  # FE_x_star[:, 0], FE_x_star[:, -1] = 0.0, 0.0
  FE_y_star[0, :], FE_y_star[-1, :] = 0.0, 0.0
  
  # Introduce block fluxes
  Frho_x_star,Frhou_x_star,Frhov_x_star,FE_x_star,Frho_y_star,Frhou_y_star,Frhov_y_star,FE_y_star = block_flux(Frho_x_star,Frhou_x_star,Frhov_x_star,FE_x_star,Frho_y_star,Frhou_y_star,Frhov_y_star,FE_y_star)

  # RK2 intervention: take final update combining F^n + F^*
  rho[Ng:-Ng, Ng:-Ng] = rho[Ng:-Ng, Ng:-Ng] \
      - 0.5 * dt/dx * (Frho_x_star[:, 1:] - Frho_x_star[:, :-1] + Frho_x[:, 1:] - Frho_x[:, :-1]) \
      - 0.5 * dt/dy * (Frho_y_star[1:, :] - Frho_y_star[:-1, :] + Frho_y[1:, :] - Frho_y[:-1, :])
  rhou[Ng:-Ng, Ng:-Ng] = rhou[Ng:-Ng, Ng:-Ng] \
      - 0.5 * dt/dx * (Frhou_x_star[:, 1:] - Frhou_x_star[:, :-1] + Frhou_x[:, 1:] - Frhou_x[:, :-1]) \
      - 0.5 * dt/dy * (Frhou_y_star[1:, :] - Frhou_y_star[:-1, :] + Frhou_y[1:, :] - Frhou_y[:-1, :])
  rhov[Ng:-Ng, Ng:-Ng] = rhov[Ng:-Ng, Ng:-Ng] \
      - 0.5 * dt/dx * (Frhov_x_star[:, 1:] - Frhov_x_star[:, :-1] + Frhov_x[:, 1:] - Frhov_x[:, :-1]) \
      - 0.5 * dt/dy * (Frhov_y_star[1:, :] - Frhov_y_star[:-1, :] + Frhov_y[1:, :] - Frhov_y[:-1, :])
  energy[Ng:-Ng, Ng:-Ng] = energy[Ng:-Ng, Ng:-Ng] \
      - 0.5 * dt/dx * (FE_x_star[:, 1:] - FE_x_star[:, :-1] + FE_x[:, 1:] - FE_x[:, :-1]) \
      - 0.5 * dt/dy * (FE_y_star[1:, :] - FE_y_star[:-1, :] + FE_y[1:, :] - FE_y[:-1, :])
#   set_trace()

  u = rhou / rho
  v = rhov / rho
  kinetic_energy = 0.5 * (rhou * rhou + rhov * rhov) / rho
  p = (energy - kinetic_energy) * (gamma - 1.0)

  ### At selected Time Steps
  if not(it % fig_skip):

    fig = plt.figure(0)

    ax  = fig.add_subplot(2,1,1)
    plt.axes(ax)
#    plt.contourf(Xc[Ng:-Ng,Ng:-Ng],Yc[Ng:-Ng,Ng:-Ng],(p[Ng:-Ng,Ng:-Ng]-np.mean(p[Ng:-Ng,Ng:-Ng]))/rho0/a0/a0)
    plt.contourf(Xc[Ng:-Ng,Ng:-Ng],Yc[Ng:-Ng,Ng:-Ng],p[Ng:-Ng,Ng:-Ng])
    plt.colorbar(label = "$p$")
    ax.set_xlim(0,Lx)
    ax.set_ylim(0,Ly)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    ax.set_title("pressure")
    
    ax  = fig.add_subplot(2,1,2)
    plt.axes(ax)
    plt.contourf(Xc[Ng:-Ng,Ng:-Ng],Yc[Ng:-Ng,Ng:-Ng],u[Ng:-Ng,Ng:-Ng])
    plt.colorbar(label = "$u$")
    ax.set_xlim(0,Lx)
    ax.set_ylim(0,Ly)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    ax.set_title("velocity")
    plt.tight_layout()
    plt.show()
