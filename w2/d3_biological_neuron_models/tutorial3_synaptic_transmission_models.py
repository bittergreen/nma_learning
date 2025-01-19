# Imports
import matplotlib.pyplot as plt
import numpy as np

from w2.d3_biological_neuron_models.utils import default_pars, Poisson_generator, my_illus_LIFSYN, my_illus_STD, \
    dynamic_syn, plot_volt_trace


def run_LIF_cond(pars, I_inj, pre_spike_train_ex, pre_spike_train_in):
    """
    Conductance-based LIF dynamics

    Args:
    pars               : parameter dictionary
    I_inj              : injected current [pA]. The injected current here
                         can be a value or an array
    pre_spike_train_ex : spike train input from presynaptic excitatory neuron
    pre_spike_train_in : spike train input from presynaptic inhibitory neuron

    Returns:
    rec_spikes         : spike times
    rec_v              : mebrane potential
    gE                 : postsynaptic excitatory conductance
    gI                 : postsynaptic inhibitory conductance

    """

    # Retrieve parameters
    V_th, V_reset = pars['V_th'], pars['V_reset']
    tau_m, g_L = pars['tau_m'], pars['g_L']
    V_init, E_L = pars['V_init'], pars['V_L']
    gE_bar, gI_bar = pars['gE_bar'], pars['gI_bar']
    VE, VI = pars['VE'], pars['VI']
    tau_syn_E, tau_syn_I = pars['tau_syn_E'], pars['tau_syn_I']
    tref = pars['tref']
    dt, range_t = pars['dt'], pars['range_t']
    Lt = range_t.size

    # Initialize
    tr = 0.
    v = np.zeros(Lt)
    v[0] = V_init
    gE = np.zeros(Lt)
    gI = np.zeros(Lt)
    Iinj = I_inj * np.ones(Lt)  # ensure Iinj has length Lt

    if pre_spike_train_ex.max() == 0:
        pre_spike_train_ex_total = np.zeros(Lt)
    else:
        pre_spike_train_ex_total = pre_spike_train_ex.sum(axis=0) * np.ones(Lt)

    if pre_spike_train_in.max() == 0:
        pre_spike_train_in_total = np.zeros(Lt)
    else:
        pre_spike_train_in_total = pre_spike_train_in.sum(axis=0) * np.ones(Lt)

    # simulation
    rec_spikes = []  # recording spike times
    for it in range(Lt - 1):
        if tr > 0:
            v[it] = V_reset
            tr = tr - 1
        elif v[it] >= V_th:  # reset voltage and record spike event
            rec_spikes.append(it)
            v[it] = V_reset
            tr = tref / dt

        # update the synaptic conductance
        gE[it + 1] = gE[it] - (dt / tau_syn_E) * gE[it] + gE_bar * pre_spike_train_ex_total[it + 1]
        gI[it + 1] = gI[it] - (dt / tau_syn_I) * gI[it] + gI_bar * pre_spike_train_in_total[it + 1]

        # calculate the increment of the membrane potential
        dv = (dt / tau_m) * (-(v[it] - E_L) \
                             - (gE[it + 1] / g_L) * (v[it] - VE) \
                             - (gI[it + 1] / g_L) * (v[it] - VI) + Iinj[it] / g_L)

        # update membrane potential
        v[it + 1] = v[it] + dv

    rec_spikes = np.array(rec_spikes) * dt

    return v, rec_spikes, gE, gI


def section1_static_synapse():
    # Get default parameters
    pars = default_pars(T=1000.)

    # Add parameters
    pars['gE_bar'] = 2.4  # [nS]
    pars['VE'] = 0.  # [mV] excitatory reversal potential
    pars['tau_syn_E'] = 2.  # [ms]
    pars['gI_bar'] = 2.4  # [nS]
    pars['VI'] = -80.  # [mV] inhibitory reversal potential
    pars['tau_syn_I'] = 5.  # [ms]

    # Generate presynaptic spike trains
    pre_spike_train_ex = Poisson_generator(pars, rate=10, n=80)
    pre_spike_train_in = Poisson_generator(pars, rate=10, n=20)

    # Simulate conductance-based LIF model
    v, rec_spikes, gE, gI = run_LIF_cond(pars, 0, pre_spike_train_ex,
                                         pre_spike_train_in)

    # Show spikes more clearly by setting voltage high
    dt, range_t = pars['dt'], pars['range_t']
    if rec_spikes.size:
        sp_num = (rec_spikes / dt).astype(int) - 1
        v[sp_num] = 10  # draw nicer spikes

    # Change the threshold
    pars['V_th'] = 1e3

    # Calculate FMP
    v_fmp, _, _, _ = run_LIF_cond(pars, 0, pre_spike_train_ex, pre_spike_train_in)

    my_illus_LIFSYN(pars, v_fmp, v)


def EI_isi_regularity(exc_rate, inh_rate):
    pars = default_pars(T=1000.)
    # Add parameters
    pars['gE_bar'] = 3.  # [nS]
    pars['VE'] = 0.  # [mV] excitatory reversal potential
    pars['tau_syn_E'] = 2.  # [ms]
    pars['gI_bar'] = 3.  # [nS]
    pars['VI'] = -80.  # [mV] inhibitory reversal potential
    pars['tau_syn_I'] = 5.  # [ms]

    pre_spike_train_ex = Poisson_generator(pars, rate=exc_rate, n=80)
    pre_spike_train_in = Poisson_generator(pars, rate=inh_rate, n=20)  # 4:1

    # Lets first simulate a neuron with identical input but with no spike
    # threshold by setting the threshold to a very high value
    # so that we can look at the free membrane potential
    pars['V_th'] = 1e3
    v_fmp, rec_spikes, gE, gI = run_LIF_cond(pars, 0, pre_spike_train_ex,
                                             pre_spike_train_in)

    # Now simulate a LIP with a regular spike threshold
    pars['V_th'] = -55.
    v, rec_spikes, gE, gI = run_LIF_cond(pars, 0, pre_spike_train_ex,
                                         pre_spike_train_in)
    dt, range_t = pars['dt'], pars['range_t']
    if rec_spikes.size:
        sp_num = (rec_spikes / dt).astype(int) - 1
        v[sp_num] = 10  # draw nicer spikes

    spike_rate = 1e3 * len(rec_spikes) / pars['T']

    cv_isi = 0.
    if len(rec_spikes) > 3:
        isi = np.diff(rec_spikes)
        cv_isi = np.std(isi) / np.mean(isi)

    print('\n')
    plt.figure(figsize=(15, 10))
    plt.subplot(211)
    plt.text(500, -35, f'Spike rate = {spike_rate:.3f} (sp/s), Mean of Free Mem Pot = {np.mean(v_fmp):.3f}',
             fontsize=16, fontweight='bold', horizontalalignment='center',
             verticalalignment='bottom')
    plt.text(500, -38.5, f'CV ISI = {cv_isi:.3f}, STD of Free Mem Pot = {np.std(v_fmp):.3f}',
             fontsize=16, fontweight='bold', horizontalalignment='center',
             verticalalignment='bottom')

    plt.plot(pars['range_t'], v_fmp, 'r', lw=1.,
             label='Free mem. pot.', zorder=2)
    plt.plot(pars['range_t'], v, 'b', lw=1.,
             label='mem. pot with spk thr', zorder=1, alpha=0.7)
    plt.axhline(pars['V_th'], 0, 1, color='k', lw=1., ls='--',
                label='Spike Threshold', zorder=1)
    plt.axhline(np.mean(v_fmp), 0, 1, color='r', lw=1., ls='--',
                label='Mean Free Mem. Pot.', zorder=1)
    plt.ylim(-76, -39)
    plt.xlabel('Time (ms)')
    plt.ylabel('V (mV)')
    plt.legend(loc=[1.02, 0.68])

    plt.subplot(223)
    plt.plot(pars['range_t'][::3], gE[::3], 'r', lw=1)
    plt.xlabel('Time (ms)')
    plt.ylabel(r'$g_E$ (nS)')

    plt.subplot(224)
    plt.plot(pars['range_t'][::3], gI[::3], 'b', lw=1)
    plt.xlabel('Time (ms)')
    plt.ylabel(r'$g_I$ (nS)')

    plt.tight_layout()
    plt.show()


# @markdown Execute for helper function for conductance-based LIF neuron with STP-synapses

def run_LIF_cond_STP(pars, I_inj, pre_spike_train_ex, pre_spike_train_in):
    """
    conductance-based LIF dynamics

    Args:
    pars               : parameter dictionary
    I_inj              : injected current [pA]
                         The injected current here can be a value or an array
    pre_spike_train_ex : spike train input from presynaptic excitatory neuron (binary)
    pre_spike_train_in : spike train input from presynaptic inhibitory neuron (binary)

    Returns:
    rec_spikes : spike times
    rec_v      : mebrane potential
    gE         : postsynaptic excitatory conductance
    gI         : postsynaptic inhibitory conductance

    """

    # Retrieve parameters
    V_th, V_reset = pars['V_th'], pars['V_reset']
    tau_m, g_L = pars['tau_m'], pars['g_L']
    V_init, V_L = pars['V_init'], pars['V_L']
    gE_bar, gI_bar = pars['gE_bar'], pars['gI_bar']
    U0E, tau_dE, tau_fE = pars['U0_E'], pars['tau_d_E'], pars['tau_f_E']
    U0I, tau_dI, tau_fI = pars['U0_I'], pars['tau_d_I'], pars['tau_f_I']
    VE, VI = pars['VE'], pars['VI']
    tau_syn_E, tau_syn_I = pars['tau_syn_E'], pars['tau_syn_I']
    tref = pars['tref']

    dt, range_t = pars['dt'], pars['range_t']
    Lt = range_t.size

    nE = pre_spike_train_ex.shape[0]
    nI = pre_spike_train_in.shape[0]

    # compute conductance Excitatory synapses
    uE = np.zeros((nE, Lt))
    RE = np.zeros((nE, Lt))
    gE = np.zeros((nE, Lt))
    for ie in range(nE):
        u, R, g = dynamic_syn(gE_bar, tau_syn_E, U0E, tau_dE, tau_fE,
                              pre_spike_train_ex[ie, :], dt)

        uE[ie, :], RE[ie, :], gE[ie, :] = u, R, g

    gE_total = gE.sum(axis=0)

    # compute conductance Inhibitory synapses
    uI = np.zeros((nI, Lt))
    RI = np.zeros((nI, Lt))
    gI = np.zeros((nI, Lt))
    for ii in range(nI):
        u, R, g = dynamic_syn(gI_bar, tau_syn_I, U0I, tau_dI, tau_fI,
                              pre_spike_train_in[ii, :], dt)

        uI[ii, :], RI[ii, :], gI[ii, :] = u, R, g

    gI_total = gI.sum(axis=0)

    # Initialize
    v = np.zeros(Lt)
    v[0] = V_init
    Iinj = I_inj * np.ones(Lt)  # ensure I has length Lt

    # simulation
    rec_spikes = []  # recording spike times
    tr = 0.
    for it in range(Lt - 1):
        if tr > 0:
            v[it] = V_reset
            tr = tr - 1
        elif v[it] >= V_th:  # reset voltage and record spike event
            rec_spikes.append(it)
            v[it] = V_reset
            tr = tref / dt

        # calculate the increment of the membrane potential
        dv = (dt / tau_m) * (-(v[it] - V_L)
                             - (gE_total[it + 1] / g_L) * (v[it] - VE)
                             - (gI_total[it + 1] / g_L) * (v[it] - VI) + Iinj[it] / g_L)

        # update membrane potential
        v[it + 1] = v[it] + dv

    rec_spikes = np.array(rec_spikes) * dt

    return v, rec_spikes, uE, RE, gE, RI, RI, gI


def LIF_STP(tau_ratio):
    pars = default_pars(T=1000)
    pars['gE_bar'] = 1.2 * 4  # [nS]
    pars['VE'] = 0.  # [mV]
    pars['tau_syn_E'] = 5.  # [ms]
    pars['gI_bar'] = 1.6 * 4  # [nS]
    pars['VI'] = -80.  # [ms]
    pars['tau_syn_I'] = 10.  # [ms]

    # here we assume that both Exc and Inh synapses have synaptic depression
    pars['U0_E'] = 0.45
    pars['tau_d_E'] = 500. * tau_ratio  # [ms]
    pars['tau_f_E'] = 300. * tau_ratio  # [ms]

    pars['U0_I'] = 0.45
    pars['tau_d_I'] = 500. * tau_ratio  # [ms]
    pars['tau_f_I'] = 300. * tau_ratio  # [ms]

    pre_spike_train_ex = Poisson_generator(pars, rate=15, n=80)
    pre_spike_train_in = Poisson_generator(pars, rate=15, n=20)  # 4:1

    v, rec_spikes, uE, RE, gE, uI, RI, gI = run_LIF_cond_STP(pars, 0,
                                                             pre_spike_train_ex,
                                                             pre_spike_train_in)

    t_plot_range = pars['range_t'] > 200

    plt.figure(figsize=(11, 7))
    plt.subplot(211)
    plot_volt_trace(pars, v, rec_spikes, show=False)

    plt.subplot(223)
    plt.plot(pars['range_t'][t_plot_range], gE.sum(axis=0)[t_plot_range], 'r')
    plt.xlabel('Time (ms)')
    plt.ylabel(r'$g_E$ (nS)')

    plt.subplot(224)
    plt.plot(pars['range_t'][t_plot_range], gI.sum(axis=0)[t_plot_range], 'b')
    plt.xlabel('Time (ms)')
    plt.ylabel(r'$g_I$ (nS)')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # section1_static_synapse()
    # EI_isi_regularity(10, 50)
    # STD
    U0 = 0.5
    tau_d = 100.
    tau_f = 50.
    # STF
    U0 = 0.2
    tau_d = 100.
    tau_f = 750.
    # _ = my_illus_STD(Poisson=True, rate=100, U0=U0, tau_d=tau_d, tau_f=tau_f)
    LIF_STP(tau_ratio=0.60)
