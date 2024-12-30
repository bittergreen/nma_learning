# @title Plotting Functions
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt, gridspec


def move_sympyplot_to_axes(p, ax):
    backend = p.backend(p)
    backend.ax = ax
    backend.process_series()
    backend.ax.spines['right'].set_color('none')
    backend.ax.spines['bottom'].set_position('zero')
    backend.ax.spines['top'].set_color('none')
    plt.close(backend.fig)


def plot_functions(function, show_derivative, show_integral):
    # For sympy we first define our symbolic variable
    x, y, z, t, f = sp.symbols('x y z t f')

    # We define our function
    if function == 'Linear':
        f = -2 * t
        name = r'$-2t$'
    elif function == 'Parabolic':
        f = t ** 2
        name = r'$t^2$'
    elif function == 'Exponential':
        f = sp.exp(t)
        name = r'$e^t$'
    elif function == 'Sine':
        f = sp.sin(t)
        name = r'$sin(t)$'
    elif function == 'Sigmoid':
        f = 1 / (1 + sp.exp(-(t - 5)))
        name = r'$\frac{1}{1+e^{-(t-5)}}$'

    if show_derivative and not show_integral:
        # Calculate the derivative of sin(t) as a function of t
        diff_f = sp.diff(f)
        print('Derivative of', f, 'is ', diff_f)

        p1 = sp.plot(f, diff_f, show=False)
        p1[0].line_color = 'r'
        p1[1].line_color = 'b'
        p1[0].label = 'Function'
        p1[1].label = 'Derivative'
        p1.legend = True
        p1.title = 'Function = ' + name + '\n'
        p1.show()
    elif show_integral and not show_derivative:

        int_f = sp.integrate(f)
        int_f = int_f - int_f.subs(t, -10)
        print('Integral of', f, 'is ', int_f)

        p1 = sp.plot(f, int_f, show=False)
        p1[0].line_color = 'r'
        p1[1].line_color = 'g'
        p1[0].label = 'Function'
        p1[1].label = 'Integral'
        p1.legend = True
        p1.title = 'Function = ' + name + '\n'
        p1.show()


    elif show_integral and show_derivative:

        diff_f = sp.diff(f)
        print('Derivative of', f, 'is ', diff_f)

        int_f = sp.integrate(f)
        int_f = int_f - int_f.subs(t, -10)
        print('Integral of', f, 'is ', int_f)

        p1 = sp.plot(f, diff_f, int_f, show=False)
        p1[0].line_color = 'r'
        p1[1].line_color = 'b'
        p1[2].line_color = 'g'
        p1[0].label = 'Function'
        p1[1].label = 'Derivative'
        p1[2].label = 'Integral'
        p1.legend = True
        p1.title = 'Function = ' + name + '\n'
        p1.show()

    else:

        p1 = sp.plot(f, show=False)
        p1[0].line_color = 'r'
        p1[0].label = 'Function'
        p1.legend = True
        p1.title = 'Function = ' + name + '\n'
        p1.show()


def plot_alpha_func(t, f, df_dt):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, f, 'r', label='Alpha function')
    plt.xlabel('Time (au)')
    plt.ylabel('Voltage')
    plt.title('Alpha function (f(t))')
    # plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, df_dt, 'b', label='Derivative')
    plt.title('Derivative of alpha function')
    plt.xlabel('Time (au)')
    plt.ylabel('df/dt')
    # plt.legend()


def plot_charge_transfer(t, PSP, numerical_integral):
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(t, PSP)
    axes[0].set(xlabel='t', ylabel='PSP')

    axes[1].plot(t, numerical_integral)
    axes[1].set(xlabel='t', ylabel='Charge Transferred')


# @title Plotting Functions

def plot_dPdt(alpha=.3):
    """ Plots change in population over time
    Args:
      alpha: Birth Rate
    Returns:
      A figure two panel figure
      left panel: change in population as a function of population
      right panel: membrane potential as a function of time
  """

    with plt.xkcd():
        time = np.arange(0, 10, 0.01)
        fig = plt.figure(figsize=(12, 4))
        gs = gridspec.GridSpec(1, 2)

        ## dpdt as a fucntion of p
        plt.subplot(gs[0])
        plt.plot(np.exp(alpha * time), alpha * np.exp(alpha * time))
        plt.xlabel(r'Population $p(t)$ (millions)')
        plt.ylabel(r'$\frac{d}{dt}p(t)=\alpha p(t)$')

        ## p exact solution
        plt.subplot(gs[1])
        plt.plot(time, np.exp(alpha * time))
        plt.ylabel(r'Population $p(t)$ (millions)')
        plt.xlabel('time (years)')
        plt.show()


def plot_V_no_input(V_reset=-75):
    """
    Args:
      V_reset: Reset Potential
    Returns:
      A figure two panel figure
      left panel: change in membrane potential as a function of membrane potential
      right panel: membrane potential as a function of time
  """
    E_L = -75
    tau_m = 10
    t = np.arange(0, 100, 0.01)
    V = E_L + (V_reset - E_L) * np.exp(-(t) / tau_m)
    V_range = np.arange(-90, 0, 1)
    dVdt = -(V_range - E_L) / tau_m

    with plt.xkcd():
        time = np.arange(0, 10, 0.01)
        fig = plt.figure(figsize=(12, 4))
        gs = gridspec.GridSpec(1, 2)

        plt.subplot(gs[0])
        plt.plot(V_range, dVdt)
        plt.hlines(0, min(V_range), max(V_range), colors='black', linestyles='dashed')
        plt.vlines(-75, min(dVdt), max(dVdt), colors='black', linestyles='dashed')
        plt.plot(V_reset, -(V_reset - E_L) / tau_m, 'o', label=r'$V_{reset}$')
        plt.text(-50, 1, 'Positive')
        plt.text(-50, -2, 'Negative')
        plt.text(E_L - 1, max(dVdt), r'$E_L$')
        plt.legend()
        plt.xlabel('Membrane Potential V (mV)')
        plt.ylabel(r'$\frac{dV}{dt}=\frac{-(V(t)-E_L)}{\tau_m}$')

        plt.subplot(gs[1])
        plt.plot(t, V)
        plt.plot(t[0], V_reset, 'o')
        plt.ylabel(r'Membrane Potential $V(t)$ (mV)')
        plt.xlabel('time (ms)')
        plt.ylim([-95, -60])

        plt.show()


## Plotting the differential Equation
def plot_dVdt(I=0):
    """
    Args:
      I  : Input Current
    Returns:
      figure of change in membrane potential as a function of membrane potential
  """

    with plt.xkcd():
        E_L = -75
        tau_m = 10
        V = np.arange(-85, 0, 1)
        g_L = 10.
        fig = plt.figure(figsize=(6, 4))

        plt.plot(V, (-(V - E_L) + I * 10) / tau_m)
        plt.hlines(0, min(V), max(V), colors='black', linestyles='dashed')
        plt.xlabel('V (mV)')
        plt.ylabel(r'$\frac{dV}{dt}$')
        plt.show()


# @title Helper Functions

## EXACT SOLUTION OF LIF
def Exact_Integrate_and_Fire(I,t):
  """
    Args:
      I  : Input Current
      t : time
    Returns:
      Spike : Spike Count
      Spike_time : Spike time
      V_exact : Exact membrane potential
  """

  Spike = 0
  tau_m = 10
  R = 10
  t_isi = 0
  V_reset = E_L = -75
  V_exact = V_reset * np.ones(len(t))
  V_th = -50
  Spike_time = []

  for i in range(0, len(t)):

    V_exact[i] = E_L + R*I + (V_reset - E_L - R*I) * np.exp(-(t[i]-t_isi)/tau_m)

    # Threshold Reset
    if V_exact[i] > V_th:
        V_exact[i-1] = 0
        V_exact[i] = V_reset
        t_isi = t[i]
        Spike = Spike+1
        Spike_time = np.append(Spike_time, t[i])

  return Spike, Spike_time, V_exact


# @title Plotting Functions

time = np.arange(0, 1, 0.01)


def plot_slope(dt):
  """
    Args:
      dt  : time-step
    Returns:
      A figure of an exponential, the slope of the exponential and the derivative exponential
  """

  t = np.arange(0, 5+0.1/2, 0.1)

  with plt.xkcd():

    fig = plt.figure(figsize=(6, 4))
    # Exponential
    p = np.exp(0.3*t)
    plt.plot(t, p, label='y')
    # slope
    plt.plot([1, 1+dt], [np.exp(0.3*1), np.exp(0.3*(1+dt))],':og',label=r'$\frac{y(1+\Delta t)-y(1)}{\Delta t}$')
    # derivative
    plt.plot([1, 1+dt], [np.exp(0.3*1), np.exp(0.3*(1))+dt*0.3*np.exp(0.3*(1))],'-k',label=r'$\frac{dy}{dt}$')
    plt.legend()
    plt.plot(1+dt, np.exp(0.3*(1+dt)), 'og')
    plt.ylabel('y')
    plt.xlabel('t')
    plt.show()


def plot_StepEuler(dt):
  """
    Args:
      dt  : time-step
    Returns:
      A figure of one step of the Euler method for an exponential growth function
  """

  t=np.arange(0, 1 + dt + 0.1 / 2, 0.1)

  with plt.xkcd():
    fig = plt.figure(figsize=(6,4))
    p=np.exp(0.3*t)
    plt.plot(t,p)
    plt.plot([1,],[np.exp(0.3*1)],'og',label='Known')
    plt.plot([1,1+dt],[np.exp(0.3*1),np.exp(0.3*(1))+dt*0.3*np.exp(0.3*1)],':g',label=r'Euler')
    plt.plot(1+dt,np.exp(0.3*(1))+dt*0.3*np.exp(0.3*1),'or',label=r'Estimate $p_1$')
    plt.plot(1+dt,p[-1],'bo',label=r'Exact $p(t_1)$')
    plt.vlines(1+dt,np.exp(0.3*(1))+dt*0.3*np.exp(0.3*1),p[-1],colors='r', linestyles='dashed',label=r'Error $e_1$')
    plt.text(1+dt+0.1,(np.exp(0.3*(1))+dt*0.3*np.exp(0.3*1)+p[-1])/2,r'$e_1$')
    plt.legend()
    plt.ylabel('Population (millions)')
    plt.xlabel('time(years)')
    plt.show()

def visualize_population_approx(t, p):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(t, np.exp(0.3*t), 'k', label='Exact Solution')

    plt.plot(t, p,':o', label='Euler Estimate')
    plt.vlines(t, p, np.exp(0.3*t),
              colors='r', linestyles='dashed', label=r'Error $e_k$')

    plt.ylabel('Population (millions)')
    plt.legend()
    plt.xlabel('Time (years)')
    plt.show()


## LIF PLOT
def plot_IF(t, V, I, Spike_time):
  """
    Args:
      t  : time
      V  : membrane Voltage
      I  : Input
      Spike_time : Spike_times
    Returns:
      figure with three panels
      top panel: Input as a function of time
      middle panel: membrane potential as a function of time
      bottom panel: Raster plot
  """

  with plt.xkcd():
    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(3, 1,  height_ratios=[1, 4, 1])
    # PLOT OF INPUT
    plt.subplot(gs[0])
    plt.ylabel(r'$I_e(nA)$')
    plt.yticks(rotation=45)
    plt.plot(t,I,'g')
    #plt.ylim((2,4))
    plt.xlim((-50,1000))
    # PLOT OF ACTIVITY
    plt.subplot(gs[1])
    plt.plot(t,V,':')
    plt.xlim((-50,1000))
    plt.ylabel(r'$V(t)$(mV)')
    # PLOT OF SPIKES
    plt.subplot(gs[2])
    plt.ylabel(r'Spike')
    plt.yticks([])
    plt.scatter(Spike_time,1*np.ones(len(Spike_time)), color="grey", marker=".")
    plt.xlim((-50,1000))
    plt.xlabel('time(ms)')
    plt.show()


def plot_rErI(t, r_E, r_I):
  """
    Args:
      t   : time
      r_E : excitation rate
      r_I : inhibition rate

    Returns:
      figure of r_I and r_E as a function of time

  """
  with plt.xkcd():
    fig = plt.figure(figsize=(6,4))
    plt.plot(t,r_E,':',color='b',label=r'$r_E$')
    plt.plot(t,r_I,':',color='r',label=r'$r_I$')
    plt.xlabel('time(ms)')
    plt.legend()
    plt.ylabel('Firing Rate (Hz)')
    plt.show()


def plot_rErI_Simple(t, r_E, r_I):
  """
    Args:
      t   : time
      r_E : excitation rate
      r_I : inhibition rate

    Returns:
      figure with two panels
      left panel: r_I and r_E as a function of time
      right panel: r_I as a function of r_E with Nullclines

  """
  with plt.xkcd():
    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(1, 2)
    # LEFT PANEL
    plt.subplot(gs[0])
    plt.plot(t,r_E,':',color='b',label=r'$r_E$')
    plt.plot(t,r_I,':',color='r',label=r'$r_I$')
    plt.xlabel('time(ms)')
    plt.legend()
    plt.ylabel('Firing Rate (Hz)')
    # RIGHT PANEL
    plt.subplot(gs[1])
    plt.plot(r_E,r_I,'k:')
    plt.plot(r_E[0],r_I[0],'go')

    plt.hlines(0,np.min(r_E),np.max(r_E),linestyles="dashed",color='b',label=r'$\frac{d}{dt}r_E=0$')
    plt.vlines(0,np.min(r_I),np.max(r_I),linestyles="dashed",color='r',label=r'$\frac{d}{dt}r_I=0$')

    plt.legend(loc='upper left')

    plt.xlabel(r'$r_E$')
    plt.ylabel(r'$r_I$')
    plt.show()


def plot_rErI_Matrix(t, r_E, r_I, Null_rE, Null_rI):
  """
    Args:
      t   : time
      r_E : excitation firing rate
      r_I : inhibition firing rate
      Null_rE: Nullclines excitation firing rate
      Null_rI: Nullclines inhibition firing rate
    Returns:
      figure with two panels
      left panel: r_I and r_E as a function of time
      right panel: r_I as a function of r_E with Nullclines

  """

  with plt.xkcd():
    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(1, 2)
    plt.subplot(gs[0])
    plt.plot(t,r_E,':',color='b',label=r'$r_E$')
    plt.plot(t,r_I,':',color='r',label=r'$r_I$')
    plt.xlabel('time(ms)')
    plt.ylabel('Firing Rate (Hz)')
    plt.legend()
    plt.subplot(gs[1])
    plt.plot(r_E,r_I,'k:')
    plt.plot(r_E[0],r_I[0],'go')

    plt.plot(r_E,Null_rE,':',color='b',label=r'$\frac{d}{dt}r_E=0$')
    plt.plot(r_E,Null_rI,':',color='r',label=r'$\frac{d}{dt}r_I=0$')
    plt.legend(loc='best')
    plt.xlabel(r'$r_E$')
    plt.ylabel(r'$r_I$')
    plt.show()

