# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp  # numerical integration solver


# @title Plotting Functions
def plot_trajectory(system, params, initial_condition, dt=0.1, T=6,
                    figtitle=None):
    """
  Shows the solution of a linear system with two variables in 3 plots.
  The first plot shows x1 over time. The second plot shows x2 over time.
  The third plot shows x1 and x2 in a phase portrait.

  Args:
    system (function): a function f(x) that computes a derivative from
                        inputs (t, [x1, x2], *params)
    params (list or tuple): list of parameters for function "system"
    initial_condition (list or array): initial condition x0
    dt (float): time step of simulation
    T (float): end time of simulation
    figtitlte (string): title for the figure

  Returns:
    nothing, but it shows a figure
  """

    # time points for which we want to evaluate solutions
    t = np.arange(0, T, dt)

    # Integrate
    # use built-in ode solver
    solution = solve_ivp(system,
                         t_span=(0, T),
                         y0=initial_condition, t_eval=t,
                         args=(params),
                         dense_output=True)
    x = solution.y

    # make a color map to visualize time
    timecolors = np.array([(1, 0, 0, i) for i in t / t[-1]])

    # make a large figure
    fig, (ah1, ah2, ah3) = plt.subplots(1, 3)
    fig.set_size_inches(10, 3)

    # plot x1 as a function of time
    ah1.scatter(t, x[0,], color=timecolors)
    ah1.set_xlabel('time')
    ah1.set_ylabel('x1', labelpad=-5)

    # plot x2 as a function of time
    ah2.scatter(t, x[1], color=timecolors)
    ah2.set_xlabel('time')
    ah2.set_ylabel('x2', labelpad=-5)

    # plot x1 and x2 in a phase portrait
    ah3.scatter(x[0,], x[1,], color=timecolors)
    ah3.set_xlabel('x1')
    ah3.set_ylabel('x2', labelpad=-5)
    # include initial condition is a blue cross
    ah3.plot(x[0, 0], x[1, 0], 'bx')

    # adjust spacing between subplots
    plt.subplots_adjust(wspace=0.5)

    # add figure title
    if figtitle is not None:
        fig.suptitle(figtitle, size=16)
    plt.show()


def plot_streamplot(A, ax, figtitle=None, show=True):
    """
  Show a stream plot for a linear ordinary differential equation with
  state vector x=[x1,x2] in axis ax.

  Args:
    A (numpy array): 2x2 matrix specifying the dynamical system
    ax (matplotlib.axes): axis to plot
    figtitle (string): title for the figure
    show (boolean): enable plt.show()

  Returns:
    nothing, but shows a figure
  """

    # sample 20 x 20 grid uniformly to get x1 and x2
    grid = np.arange(-20, 21, 1)
    x1, x2 = np.meshgrid(grid, grid)

    # calculate x1dot and x2dot at each grid point
    x1dot = A[0, 0] * x1 + A[0, 1] * x2
    x2dot = A[1, 0] * x1 + A[1, 1] * x2

    # make a colormap
    magnitude = np.sqrt(x1dot ** 2 + x2dot ** 2)
    color = 2 * np.log1p(magnitude)  # Avoid taking log of zero

    # plot
    plt.sca(ax)
    plt.streamplot(x1, x2, x1dot, x2dot, color=color,
                   linewidth=1, cmap=plt.cm.cividis,
                   density=2, arrowstyle='->', arrowsize=1.5)
    plt.xlabel(r'$x1$')
    plt.ylabel(r'$x2$')

    # figure title
    if figtitle is not None:
        plt.title(figtitle, size=16)

    # include eigenvectors
    if True:
        # get eigenvalues and eigenvectors of A
        lam, v = np.linalg.eig(A)

        # get eigenvectors of A
        eigenvector1 = v[:, 0].real
        eigenvector2 = v[:, 1].real

        # plot eigenvectors
        plt.arrow(0, 0, 20 * eigenvector1[0], 20 * eigenvector1[1],
                  width=0.5, color='r', head_width=2,
                  length_includes_head=True)
        plt.arrow(0, 0, 20 * eigenvector2[0], 20 * eigenvector2[1],
                  width=0.5, color='b', head_width=2,
                  length_includes_head=True)
    if show:
        plt.show()


def plot_specific_example_stream_plots(A_options):
    """
  Show a stream plot for each A in A_options

  Args:
    A (list): a list of numpy arrays (each element is A)

  Returns:
    nothing, but shows a figure
  """
    # get stream plots for the four different systems
    plt.figure(figsize=(10, 10))

    for i, A in enumerate(A_options):

        ax = plt.subplot(2, 2, 1 + i)
        # get eigenvalues and eigenvectors
        lam, v = np.linalg.eig(A)

        # plot eigenvalues as title
        # (two spaces looks better than one)
        eigstr = ",  ".join([f"{x:.2f}" for x in lam])
        figtitle = f"A with eigenvalues\n" + '[' + eigstr + ']'
        plot_streamplot(A, ax, figtitle=figtitle, show=False)

        # Remove y_labels on righthand plots
        if i % 2:
            ax.set_ylabel(None)
        if i < 2:
            ax.set_xlabel(None)

        plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()


# @title Plotting Functions

def plot_switch_simulation(t, x):
    plt.figure()
    plt.plot(t, x)
    plt.title('State-switch simulation')
    plt.xlabel('Time')
    plt.xlim((0, 300))  # zoom in time
    plt.ylabel('State of ion channel 0/1', labelpad=-60)
    plt.yticks([0, 1], ['Closed (0)', 'Open (1)'])
    plt.show()


def plot_interswitch_interval_histogram(inter_switch_intervals):
    plt.figure()
    plt.hist(inter_switch_intervals)
    plt.title('Inter-switch Intervals Distribution')
    plt.ylabel('Interval Count')
    plt.xlabel('time')
    plt.show()


def plot_state_probabilities(time, states):
    plt.figure()
    plt.plot(time, states[:, 0], label='Closed')
    plt.plot(time, states[:, 1], label='Open')
    plt.xlabel('time')
    plt.ylabel('prob(open OR closed)')
    plt.legend()
    plt.show()


# @title Plotting Functions

def plot_random_walk_sims(sims, nsims=10):
    """Helper for exercise 3A"""
    plt.figure()
    plt.plot(sims[:nsims, :].T)
    plt.xlabel('time')
    plt.ylabel('position x')
    plt.show()


def plot_mean_var_by_timestep(mu, var):
    """Helper function for exercise 3A.2"""
    fig, (ah1, ah2) = plt.subplots(2)

    # plot mean of distribution as a function of time
    ah1.plot(mu)
    ah1.set(ylabel='mean')
    ah1.set_ylim([-5, 5])

    # plot variance of distribution as a function of time
    ah2.plot(var)
    ah2.set(xlabel='time')
    ah2.set(ylabel='variance')

    plt.show()


def plot_ddm(t, x, xinfty, lam, x0):
    plt.figure()
    plt.plot(t, xinfty * (1 - lam ** t) + x0 * lam ** t, 'r',
             label='deterministic solution')
    plt.plot(t, x, 'k.', label='simulation')  # simulated data pts
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    plt.show()


def var_comparison_plot(empirical, analytical):
    plt.figure()
    plt.plot(empirical, analytical, '.', markersize=15)
    plt.xlabel('empirical equilibrium variance')
    plt.ylabel('analytic equilibrium variance')
    plt.plot(np.arange(8), np.arange(8), 'k', label='45 deg line')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_dynamics(x, t, lam, xinfty=0):
    """ Plot the dynamics """
    plt.figure()
    plt.title('$\lambda=%0.1f$' % lam, fontsize=16)
    x0 = x[0]
    plt.plot(t, xinfty + (x0 - xinfty) * lam ** t, 'r', label='analytic solution')
    plt.plot(t, x, 'k.', label='simulation')  # simulated data pts
    plt.ylim(0, x0 + 1)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    plt.show()


# @title Plotting Functions

def plot_residual_histogram(res):
    """Helper function for Exercise 4A"""
    plt.figure()
    plt.hist(res)
    plt.xlabel('error in linear model')
    plt.title(f'stdev of errors = {res.std():.4f}')
    plt.show()


def plot_training_fit(x1, x2, p, r):
    """Helper function for Exercise 4B"""
    plt.figure()
    plt.scatter(x2 + np.random.standard_normal(len(x2)) * 0.02,
                np.dot(x1.T, p), alpha=0.2)
    plt.title(f'Training fit, order {r} AR model')
    plt.xlabel('x')
    plt.ylabel('estimated x')
    plt.show()


# @title Helper Functions

def ddm(T, x0, xinfty, lam, sig):
    '''
  Samples a trajectory of a drift-diffusion model.

  args:
  T (integer): length of time of the trajectory
  x0 (float): position at time 0
  xinfty (float): equilibrium position
  lam (float): process param
  sig: standard deviation of the normal distribution

  returns:
  t (numpy array of floats): time steps from 0 to T sampled every 1 unit
  x (numpy array of floats): position at every time step
  '''
    t = np.arange(0, T, 1.)
    x = np.zeros_like(t)
    x[0] = x0

    for k in range(len(t) - 1):
        x[k + 1] = xinfty + lam * (x[k] - xinfty) + sig * np.random.standard_normal(size=1)

    return t, x


def build_time_delay_matrices(x, r):
    """
    Builds x1 and x2 for regression

    Args:
    x (numpy array of floats): data to be auto regressed
    r (scalar): order of Autoregression model

    Returns:
    (numpy array of floats) : to predict "x2"
    (numpy array of floats) : predictors of size [r,n-r], "x1"

    """
    # construct the time-delayed data matrices for order-r AR model
    x1 = np.ones(len(x) - r)
    x1 = np.vstack((x1, x[0:-r]))
    xprime = x
    for i in range(r - 1):
        xprime = np.roll(xprime, -1)
        x1 = np.vstack((x1, xprime[0:-r]))

    x2 = x[r:]

    return x1, x2


def AR_prediction(x_test, p):
    """
    Returns the prediction for test data "x_test" with the regression
    coefficients p

    Args:
    x_test (numpy array of floats): test data to be predicted
    p (numpy array of floats): regression coefficients of size [r] after
    solving the autoregression (order r) problem on train data

    Returns:
    (numpy array of floats): Predictions for test data. +1 if positive and -1
    if negative.
    """
    x1, x2 = build_time_delay_matrices(x_test, len(p) - 1)

    # Evaluating the AR_model function fit returns a number.
    # We take the sign (- or +) of this number as the model's guess.
    return np.sign(np.dot(x1.T, p))


def error_rate(x_test, p):
    """
    Returns the error of the Autoregression model. Error is the number of
    mismatched predictions divided by total number of test points.

    Args:
    x_test (numpy array of floats): data to be predicted
    p (numpy array of floats): regression coefficients of size [r] after
    solving the autoregression (order r) problem on train data

    Returns:
    (float): Error (percentage).
    """
    x1, x2 = build_time_delay_matrices(x_test, len(p) - 1)

    return np.count_nonzero(x2 - AR_prediction(x_test, p)) / len(x2)

