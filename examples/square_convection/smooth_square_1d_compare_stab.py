"""
This small example presents modelling of a square function (smoothed at the base).
Theoretically, the initial conditions should be found back after each period.  So it is used to verify and compare the accuracy of the different stabilisation methods. 
"""
from dolfin import *
import numpy as np
from timeit import default_timer as timer
import time
import sys
from fedm.physical_constants import *
from fedm.file_io import *
from fedm.functions import *

# Choose type of stabilisation
is_LOG = False
is_SUPG = True
is_Bubble = False

# Optimization parameters.
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters['krylov_solver']['nonzero_initial_guess'] = True

# Defining tye of used solver and its parameters.
linear_solver = "mumps" # Setting linear solver: mumps | gmres | lu
if is_LOG:
    maximum_iterations = 400 # Setting up maximum number of nonlinear solver iterations
else:
    maximum_iterations = 1 # Setting up maximum number of nonlinear solver iterations

relative_tolerance = 1e-3 # Setting up relative tolerance



# ============================================================================
# Definition of the simulation conditions, model and coordinates
# ============================================================================
model = 'Time_of_flight' # Model name
coordinates = 'cylindrical' # Coordinates choice
gas = 'Air'
Tgas = 300.0 # Gas temperature in [K]
p0 = 760.0 # Pressure in [Torr]
N0 = p0*3.21877e22 # Number density in [m^-3]


# ============================================================================
# Defining number of species for which the problem is solved, their properties and creating output files.
# ============================================================================
number_of_species = 1

if is_LOG:
    elec_name = 'electrons_log'
elif is_SUPG:
    elec_name = 'electrons_supg'
elif is_Bubble:
    elec_name = 'electrons_bubble'
else:
    elec_name = 'electrons'

particle_species_type = [elec_name, 'analytical solution'] # Defining particle species types, where the values are used as names for output files
M = me
charge = -elementary_charge
equation_type = ['drift-diffusion-reaction'] # Defining the type of the equation (reaction | diffusion-reaction | drift-diffusion-reaction)
wez = Expression('1 + 9*pow(sin(pi*x[0]*1e+3), 8)', degree = 1) #1.7e5 # Electron drift velocity z component [m/s] 
we = Expression(('1 + 9*pow(sin(pi*x[0]*1e+3), 8)',), degree = 1) 
De = 0. # Electron Diffusion coefficient [m^2/s]
alpha_e = 0. # Effective ionization coefficient [1/m]
x0 =0.15e-3
l = 0.00004 # 0.04 # Gaussian characteristic width

log('properties', files.model_log, gas, model, particle_species_type, M, charge) # Writting particle properties into a log file
vtkfile_u = output_files('pvd', 'number density', particle_species_type) # Setting-up output files

# ============================================================================
# Definition of the time variables.
# ============================================================================
t_old = None # Previous time step
t0 = 0 # Initial time step
t = t0 # Current time step
T_final = 0.591*1e-3 ; #2*5.88235294e-9 # Simulation time end [s]

dt_init = T_final/400 # Initial time step size
dt = Expression("time_step", time_step = dt_init, degree = 0) # Time step size [s]
dt_old = Expression("time_step", time_step = 1e30, degree = 0) # Time step size [s] set up as a large value to reduce the order of BDF.

t_output_step = dt_init #0.5e-9 # Time step intervals at which the results are printed to file
t_output = t0 #+ 10*dt_init # Initial output time step


# ============================================================================
# Defining the geometry of the problem and corresponding boundaries.
# ============================================================================
if coordinates == 'cylindrical':
    #r = Expression('x[0]', degree = 1)
    z = Expression('x[0]', degree = 1)

# Gap length and width
box_height = 1e-3 # [m]

boundaries = [['point', 0.0, 0.0],\
                ['point', 0.0, box_height]] # Defining list of boundaries lying between z0 and z1

class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(near(x[0], 0) and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - box_height
        #y[1] = x[1]

# ============================================================================
# Mesh setup. Structured mesh is generated using built-in mesh generator.
# ============================================================================
mesh = IntervalMesh(1000, 0. , box_height) # Generating structured mesh.
mesh_statistics(mesh) # Prints number of elements, minimal and maximal cell diameter.
h = MPI.max(MPI.comm_world, mesh.hmax()) # Maximuml cell size in mesh.

log('conditions', files.model_log, dt.time_step, 'None', p0, box_height, N0, Tgas)
log('initial time', files.model_log, t)

# ============================================================================
# Defining type of elements and function space, test functions, trial functions and functions for storing variables, and weak form
# of equation.
# ============================================================================
#pbc = PeriodicBoundary()
#V = FunctionSpace(mesh, 'P', 2)
if is_Bubble:
    QE = FiniteElement(mesh, "CG", 1)
    BE = FiniteElement(mesh, 'B', 2)
    V = FunctionSpace(mesh,QE+BE, constrained_domain=PeriodicBoundary())
    # Not working : *** FFC warning: evaluate_dof(s) for enriched element not implemented.
else:
    V = FunctionSpace(mesh, 'CG', 1, constrained_domain=PeriodicBoundary()) # Defining function space

W = VectorFunctionSpace(mesh, 'CG', 1, constrained_domain=PeriodicBoundary()) # Defining vector function space


u = TrialFunction(V) # Defining trial function
v = TestFunction(V) # Defining test function
u_old = Function(V) # Defining function for storing the data at k-1 time step
u_old1 = Function(V) # Defining function for storing the data at k-2 time step
u_new = Function(V) # Defining function for storing the data at k time step

if is_LOG:
    u_analytical  = Expression('(x[0] >= (x0 - 0.03e-3)) & (x[0] <= (x0 + 0.03e-3)) ? std::log(1.0) : std::log(exp(-(pow((x[0] - x0)/l, 2))) + DOLFIN_EPS)', x0=x0, l=l, degree=3) # Analytical solution of the particle balance equation.
else:
    u_analytical  = Expression('(x[0] >= (x0 - 0.03e-3)) & (x[0] <= (x0 + 0.03e-3)) ? 1.0 : exp(-(pow((x[0] - x0)/l, 2)))', x0=x0, l=l, element=V.ufl_element()) # Analytical solution of the particle balance equation.

u_old.assign(interpolate(u_analytical , V)) # Setting up value at k-1 time step
u_old1.assign(interpolate(u_analytical , V)) # Setting up value at k-2 time step
u_new.assign(interpolate(u_analytical, V))

w = interpolate(we, W)
D = interpolate(Constant(De), V) # Diffusion coefficient [m^2/s]
alpha_eff = interpolate(Constant(alpha_e), V) #Effective ionization coefficient [1/m]

#Gamma = -grad(D*exp(u)) + w*exp(u) # Defining electron flux [m^{-2} s^{-1}]
if is_LOG:
    Gamma = -D*grad(exp(u)) + w*exp(u) # Defining electron flux [m^{-2} s^{-1}]
    Gamma_old = -D*grad(exp(u_old)) + w*exp(u_old) # Defining electron flux [m^{-2} s^{-1}]
else:
    Gamma = -D*grad(u) + w*u # Defining electron flux [m^{-2} s^{-1}]
    Gamma_old = -D*grad(u_old) + w*u_old # Defining electron flux [m^{-2} s^{-1}]

#f = Expression('exp(-(pow((x[0] - x0 -w*t)/l, 2))/(1 + 4.0*D*t/pow(l,2))+alpha*w*t)*(w*alpha)/pow(1 + 4*D*t/pow(l,2),0.5)', x0=x0, D = De, w = wez, alpha = alpha_e, t = t, pi=pi, l=l,  degree = 2) # Defining source term
f = Expression('DOLFIN_EPS',  degree = 2) # Defining source term

theta = 1.
if is_LOG:
    F = weak_form_balance_equation_log_representation(equation_type[0], dt, dt_old, dx, u, u_old, u_old1, v, f, Gamma, is_theta_scheme=True, theta=theta, Gamma_old=Gamma_old, h=h)
else:
    F = weak_form_balance_equation(equation_type[0], dt, dt_old, dx, u, u_old, u_old1, v, f, Gamma, is_theta_scheme=True, theta=theta, Gamma_old=Gamma_old, h=h) # Definition of variational formulation of the balance equation for the electrons

    if is_SUPG:
        tauwgradv = ((h/(2*wez))*inner(w, grad(v)))
        Fsupg = weak_form_supg_balance_equation(equation_type[0], dt, dt_old, dx, u, u_old, u_old1, tauwgradv, f, Gamma, is_theta_scheme=True, theta=theta, Gamma_old=Gamma_old, h=h) # Definition of variational formulation of the balance equation for the electrons
        F = F + Fsupg


# ============================================================================
# Setting-up nonlinear solver
# ============================================================================
# Defining the problem
F = action(F, u_new)
J = derivative(F, u_new, u)
problem = Problem(J, F, [])

# Initializing nonlinear solver and setting up the parameters
nonlinear_solver = PETScSNESSolver() # Nonlinear solver initialization
nonlinear_solver.parameters['relative_tolerance'] = relative_tolerance # Setting up relative tolerance of the nonlinear solver
nonlinear_solver.parameters["linear_solver"]= linear_solver # Setting up linear solver
nonlinear_solver.parameters['maximum_iterations'] = maximum_iterations # Setting up maximum number of iterations
# nonlinear_solver.parameters["preconditioner"]="hypre_amg" # Setting the preconditioner, uncomment if iterative solver is used

n_exact = Function(V) # Defining function for storing the data at k-2 time step
n_num = Function(V) # Defining function for storing the data at k time step

while abs(t-T_final)/T_final > 1e-6:

    if abs(t-t_output) <= 1e-6*t_output:

        if is_LOG:
            n_exact.assign(project(exp(u_analytical), V, solver_type='mumps'))
            n_num.assign(project(exp(u_new), V, solver_type='mumps'))
        else:
            n_exact.assign(project((u_analytical), V, solver_type='mumps'))
            n_num.assign(project((u_new), V, solver_type='mumps'))

        relative_error = errornorm(n_num, n_exact, 'l2')/norm(n_exact, 'l2') # Calculating relative difference between exact analytical and numerical solution
        with open(files.error_file, "a") as f_err:
            f_err.write('h_max = ' + str(h) + '\t dt = ' + str(dt.time_step) + '\t relative_error = ' + str(relative_error) + '\n')
        if(MPI.rank(MPI.comm_world)==0):
            print(relative_error)
        vtkfile_u[0] <<  (n_num, t)
        vtkfile_u[1] << (n_exact, t)
        t_output += t_output_step
        print("Saved")

    t_old = t # Updating old time steps
    u_old1.assign(u_old) # Updating variable value in k-2 time step
    u_old.assign(u_new) # Updating variable value in k-1 time step
    t += dt.time_step # Updating the new  time steps

    log('time', files.model_log, t) # Time logging
    print_time(t) # Printing out current time step

    f.t=t # Updating the source term for the current time step
    u_analytical.t = t # Updating the analytical solution for the current time step

    nonlinear_solver.solve(problem, u_new.vector()) # Solving the system of equations
    #solve(F, u_new, [])

if t > (t0 + dt_init):
    dt_old.time_step = dt.time_step # For initialization BDF1 is used and after initial step BDF2 is activated
    if(MPI.rank(MPI.comm_world)==0):
        print(str(dt_old.time_step) + '\t' + str(dt.time_step) +'\n')

        if is_LOG:
            n_exact.assign(project(exp(u_analytical), V, solver_type='mumps'))
            n_num.assign(project(exp(u_new), V, solver_type='mumps'))
        else:
            n_exact.assign(project((u_analytical), V, solver_type='mumps'))
            n_num.assign(project((u_new), V, solver_type='mumps'))

    relative_error = errornorm(n_num, n_exact, 'l2')/norm(n_exact, 'l2') # Calculating relative difference between exact analytical and numerical solution
    with open(files.error_file, "a") as f_err:
        f_err.write('h_max = ' + str(h) + '\t dt = ' + str(dt.time_step) + '\t relative_error = ' + str(relative_error) + '\n')
    if(MPI.rank(MPI.comm_world)==0):
        print(relative_error)
    vtkfile_u[0] <<  (n_num, t)
    vtkfile_u[1] << (n_exact, t)
    t_output += t_output_step
    print("Saved")

print("Finished")