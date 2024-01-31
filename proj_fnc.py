import scipy
from scipy.stats import linregress
from scipy.integrate import simpson
import numpy as np
from numpy import log10
import matplotlib.pyplot as plt
from math import ceil
from inspect import isfunction
from sys import getsizeof

def signif(x, p): 
    """Writes x with p number of significant digits"""
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

def calc_rel_error(numerical, analytic, errortype="infty"):
    """Calculates the relative difference between numerical and analytic with the norm given by errortype."""
    if errortype=="infty":
        return np.max(np.abs(numerical-analytic))/np.max(np.abs(analytic))
    elif errortype=="L1":
        return simpson(np.abs(numerical-analytic))/simpson(np.abs(analytic))
        #return np.sum(np.abs(numerical-analytic))/np.sum(np.abs(analytic))
    else:
        raise Exception("Only errortypes 'infty' and 'L1' are supported.")

def loglog(solver, solution, N_xs, arguments, first_reg_node=0, last_reg_node=-1, errortype="infty", verbose=1, title=None, filename=None):
    """Makes a loglog plot.
    
    Parameters
    ----------
    solver: function
        Function for solving the problem.
    arguments: Dict
        Dictionary containing input data for the solver.
    N_xs: list
        List with number of spatial steps to be considered.
    first_reg_node: int, optional
        Regression will be done on the error points [first_reg_node,last_reg_node)
    last_reg_node: int, optional
        Regression will be done on the error points [first_reg_node,last_reg_node)
    errortype: string, optional
        "infty" or "L1", determining which error term to use.
    verbose: int, optional
        0: no printing, >0: print information continuously
    title: string, optional
        If given, the plot will be given this title.
    filename: string, optional
        If given, the plot will be saved as a pdf with this name.
    """
    N_xs = np.array(N_xs)
    dxs=1/(N_xs+1)
    errors=[]

    for N_x in N_xs:
        num, div = solver(N_x, arguments, verbose=verbose)
        x,t = div["x"], div["t"]
        if solver.__name__=="solve_HJ":
            numerical = num[:,-1]
            analytic  = solution(x,t[-1])
        elif solver.__name__=="solve_FP":
            numerical = num[:,0]
            analytic  = solution(x,t[0])
        errors.append(calc_rel_error(numerical, analytic, errortype))

        if verbose>0: 
            print("The relative error is: {:.2e}".format(errors[-1]))
            print(f"Finished N_x={N_x}\n")

    error_name = "l^\infty" if errortype=="infty" else "L^1"

    plt.figure(figsize=(6, 4.5))

    #Interpolation for numerical convergence rate
    res = linregress(log10(dxs[first_reg_node:last_reg_node]), log10(errors[first_reg_node:last_reg_node]))
    print(f"Numerical convergence rate: {signif(res.slope,3)}")
    slope = 0.5*round(res.slope/0.5) #Plot exponential slope rounded to nearest half.
    plt.loglog(dxs, 10**(res.intercept + slope*log10(dxs)),label="$\Delta x^{"+str(slope)+"}$",linestyle="--")

    #Plotting observed errors
    plt.loglog(dxs, errors, linestyle='', marker='o', label="Numerical results")
    plt.gca().invert_xaxis() #Inverting the x-axis and thus having largest stepsizes first
    if title: plt.title(title)
    plt.xlabel("Steplength $\Delta x$")
    ylabel = "Relative $"+f"{error_name}$ error"
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    if filename: plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()

def _make_xt(dx,dt,N_x,N_t):
    x = np.array([dx*(i+1/2) for i in range(N_x+1)])
    t = np.array([dt*i for i in range(N_t+1)])
    xt = np.array([[(dx*(j+1/2),dt*i) for i in range(N_t+1)] for j in range(N_x+1)])
    return {"x":x,"t":t,"xt":xt}

def make_grid(N_x, t_end, FP_lamda=None, HJ_theta=None):
    """Makes a grid in space and time satisfying the CFL condition of the Hamilton Jacobi equation or choose dt=lamda*dx for the Fokker Planck equation."""
    dx = 1/(N_x+1)
    if HJ_theta:
        dt_temp = dx*0.9/(2*HJ_theta) #Upper bound dx/(2*theta) for HJ
    elif FP_lamda:
        dt_temp = FP_lamda*dx
    else:
        dt_temp=dx

    N_t = ceil(t_end/dt_temp) #To get last value of discretized t on t_end
    dt = t_end/N_t

    return {"dx":dx, "dt":dt, "N_t":N_t, **_make_xt(dx,dt,N_x,N_t)}

def forward_diff(f, t, xt, dx, N_t, boundary):
    if isfunction(f):
        f = f(xt[:,:,0], xt[:,:,1])

    eta0, eta1 = boundary(t)

    D_plus  = 1/dx * np.array([(np.roll(f[:,i],-1) - f[:,i]) for i in range(N_t+1)]).T
    D_plus[-1,:]=eta1

    return D_plus

def backward_diff(f, t, xt, dx, N_t, boundary):
    """Calculates the derivative of f using backward difference."""
    if isfunction(f):
        f = f(xt[:,:,0], xt[:,:,1])

    eta0, eta1 = boundary(t)

    D_minus = 1/dx * np.array([(f[:,i]- np.roll(f[:,i],1)) for i in range(N_t+1)]).T
    D_minus[0,:]=eta0
    
    return D_minus

def central_diff(f, t, xt, dx, N_t, boundary):
    """Calculates the derivative of f using central differences."""
    return (forward_diff(f, t, xt, dx, N_t, boundary) + backward_diff(f, t, xt, dx, N_t, boundary))/2

def solve_HJ(N_x, arguments, verbose=0):
    """Solves the Hamilton-Jacobi equation for H(q)=q^2 with the Lax-Friedrich method.
    
    Parameters
    -----------
    N_x: int
        Divides the x-axis into N_x+1 intervals
    arguments: Dict
        b: float
            The diffusion coefficient
        mu: function or np.array
            From the Hamilton Jacobi equation.
        f: function or np.array
            From the Hamilton Jacobi equation.
        t_end: float
            The largest t-value.
        R: float
            upper bound of the difference approximation of |u_x|
        theta: float, optional
            Constant used in Lax-Friedrich. Chosen as R if not given.
        initial: function
            u(x,0), which is the initial condition for the HJ-equation
        boundary: function
            The neumann boundary condition as a function that returns the tuple (u_x(0,t),u_x(1,t)).
    verbose: int, optional
        0: no print, >0: larger number means more printing
    ret: bool, optional
        Needs to be true for the function

    Returns
    -------
    U: np.array
        The numerical solution
    : Dict
        Some useful variables constructed by the function
    """

    b = arguments["b"]
    t_end = arguments["t_end"]
    R = arguments["R"]
    theta = arguments["theta"] if "theta" in arguments.keys() else R

    mu = arguments["mu"]
    f = arguments["f"]
    initial = arguments["initial"]
    boundary = arguments["boundary"]
    
    #OBS makes mu and f to be ndarray
    if "xt" not in arguments.keys(): #
        dx,dt,N_t,x,t,xt = make_grid(N_x, t_end, HJ_theta=theta).values()
        mu = mu(xt[:,:,0], xt[:,:,1])
        f  = f(xt[:,:,0], xt[:,:,1])
    else:
        N_x, N_t = arguments["N_x"], arguments["N_t"]
        dx, dt = arguments["dx"], arguments["dt"]
        x, t = arguments["x"], arguments["t"]
        xt = arguments["xt"]

        if dt/dx > 1/(2*theta): print("delta t/delta x is probably too large.")

    #Some constants
    lamda = dt/dx
    r = dt/dx**2
    sigma  = b*r

    #Initializations
    U=np.zeros((N_x+1,N_t+1))
    U[:,0] = initial(x)
    boundary_cond_term = np.zeros(N_x+1)

    if verbose > 0: print(f"Approximate memory usage of solution U: {getsizeof(U)/10**6} MB")

    #Make constant part of A
    upper  = -sigma*np.ones(N_x+1)
    lower  = -sigma*np.ones(N_x+1)
    middle = (1+2*sigma)*np.ones(N_x+1)
    middle[[0,-1]] -= sigma
    A = scipy.sparse.spdiags([upper, middle, lower],[1,0,-1])

    eta0_old, eta1_old = boundary(t[0])
    for n in range(N_t):
        eta0, eta1 = boundary(t[n+1]) #Get neumann boundary conditions
        
        A.setdiag(middle + dt*mu[:,n+1], 0) #Update the time-dependent diagonal of A
        
        #Forward difference and backwards difference vectors for use in the numerical Hamiltonian
        D_minus_U    = (U[:,n]- np.roll(U[:,n],1))/dx
        D_minus_U[0] = eta0_old

        D_plus_U     = (np.roll(U[:,n],-1) - U[:,n])/dx
        D_plus_U[-1] = eta1_old

        boundary_cond_term[[0,-1]] = b*lamda*np.array([-eta0, eta1])
        
        g = ( (D_plus_U+D_minus_U) /2 )**2 - theta*(D_plus_U-D_minus_U)

        U[:,n+1] = scipy.sparse.linalg.bicgstab(
            A, U[:,n] + boundary_cond_term - dt*f[:,n+1] -dt*g
        )[0]

        eta0_old, eta1_old = eta0, eta1

    if verbose >1 : #Print interesting statistics
        print(f"Max value:{signif(np.max(U),3)} \nMin value:{signif(np.min(U),3)} \nMean value:{signif(np.mean(U),3)} \nStandard deviation:{signif(np.std(U),3)}\n")

    return U, {"x":x, "t":t, "xt":xt, "dt":dt, "dx":dx, "N_x":N_x, 
                   "N_t":N_t, "t_end":t_end, "b":b, "R":R, "theta":theta, 
                   "f":f, "mu":mu, "initial":initial, "boundary":boundary}

def solve_FP(N_x, arguments, verbose=0):
    """Solves the Fokker-Planck equation with a no-flux boundary condition in space and terminal condition in time, with H(q)=q^2 and utilizing the Lax-Friedrich method.
    
    Parameters
    -----------
    N_x: int
        Divides the x-axis into N_x+1 intervals
    arguments: Dict
        b: float
            The diffusion coefficient
        u: function or np.array
            From the Fokker-Planck equation.
        boundary_u: function
            The neumann boundary condition in space as a function that returns the tuple (u_x(0,t),u_x(1,t)).
        h: function or np.array
            The right hand side of the Fokker-Planck equation. h=m_t+(2m u_x + bm_x))_x
        t_end: float
            The largest t-value.
        R: float
            upper bound of the difference approximation of |u_x|
        theta: float, optional
            Constant used in Lax-Friedrich. Chosen as R if not given.
        lamda: float, optional
            Chooses time discretization. dt=lamda*dx. OBS, will be slightly altered due to wanting dt*N_t=t_end.
        terminal: function
            m(x,t_end), which is the terminal condition for the Fokker-Planck equation
        boundary: function
            The neumann boundary condition on u as a function that returns the tuple (u_x(0,t),u_x(1,t)).
    verbose: int, optional
        0: no print, >0: larger number means more printing

    Returns
    -------
    M: np.array
        The numerical solution
    : Dict
        Some useful variables constructed by the function
    """

    b = arguments["b"]
    t_end = arguments["t_end"]
    R = arguments["R"]
    theta = arguments["theta"] if "theta" in arguments.keys() else R

    terminal = arguments["terminal"]
    h = arguments["h"]
    u = arguments["u"]
    boundary_u = arguments["boundary_u"]
    
    
    #OBS makes u and f to be ndarray
    if "xt" not in arguments.keys():
        lamda = arguments["lamda"] if "lamda" in arguments.keys() else 1
        dx,dt,N_t,x,t,xt = make_grid(N_x, t_end, FP_lamda=lamda).values()
        u = u(xt[:,:,0], xt[:,:,1])
        h = h(xt[:,:,0], xt[:,:,1])

        #Find central difference for u_x and h_x
        u_x_central = central_diff(u, t, xt, dx, N_t, boundary_u)

    else:
        N_x, N_t = arguments["N_x"], arguments["N_t"]
        dx, dt = arguments["dx"], arguments["dt"]
        x, t = arguments["x"], arguments["t"]
        xt = arguments["xt"]
        u_x_central = arguments["u_x_central"]
    
    #Some constants
    lamda = dt/dx
    phi= b/dx +theta

    #Initializations
    M = np.zeros((N_x+1,N_t+1))
    M[:,-1] = terminal(x)
    l_middle = np.zeros(N_x+1)

    #Make constant part of A
    upper  = -lamda*phi*np.ones(N_x+1)
    lower  = -lamda*phi*np.ones(N_x+1)
    middle = (1+2*lamda*phi)*np.ones(N_x+1)
    middle[[0,-1]] -= lamda*phi
    A = scipy.sparse.spdiags([upper, middle, lower],[1,0,-1])

    if verbose > 0: print(f"Approximate memory usage of solution M: {getsizeof(M)/10**6} MB")
    if verbose > 1: print(f"Max of central diff approx of u derivative: {signif(np.max(np.abs(u_x_central)),3)}")
    
    for n in range(N_t,0,-1):

        #Changeing the time-dependent parts of A
        l = u_x_central[:,n-1]
        l_middle[0], l_middle[-1] = -l[0], l[-1]
        
        A.setdiag(upper - lamda*np.roll(l,-1), 1)
        A.setdiag(middle+ lamda*l_middle, 0)
        A.setdiag(lower + lamda*l, -1)

        M[:,n-1]=scipy.sparse.linalg.bicgstab(A, M[:,n] - dt*h[:,n-1] )[0]


    if verbose > 1: #Print interesting statistics
        print(f"Max value:{signif(np.max(M),3)} \nMin value:{signif(np.min(M),3)} \nMean value:{signif(np.mean(M),3)} \nStandard deviation:{signif(np.std(M),3)}\n")
    
    return M, {"x":x, "t":t, "xt":xt, "dt":dt, "dx":dx, "N_x":N_x, 
                   "N_t":N_t, "t_end":t_end, "b":b, "R":R, "theta":theta,
                    "u_x_central":u_x_central, "h":h, "terminal":terminal}
