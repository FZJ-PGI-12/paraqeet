Arbitrary bosonic state preparation using smooth pulses : Gradient based dCRAB optimization using GOAT over GRAPE method
========================================================================================================================

In this example we solve the task in the previous example of preparing
arbitrary bosonic state, but using smooth functions as the basis. Here
we perform a gradient based dCRAB optimization, by using the
*GOAToverGRAPE* (or the GROUP [Sørensen2018]) method.

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    
    from paraqeet.file_logger import FileLogger
    from paraqeet.measurement.goat_over_grape import GOATOverGRAPE
    from paraqeet.measurement.state_transfer_fidelity import StateTransferFidelityGRAPE
    from paraqeet.measurement.weighted_sum_goal import WeightedSumGoal
    from paraqeet.model.closed_system import ClosedSystem
    from paraqeet.model.composite_hamiltonian import CompositeHamiltonian
    from paraqeet.model.coupling import TwoBodyCoupling
    from paraqeet.model.qubit import Qubit
    from paraqeet.model.resonator import Resonator
    from paraqeet.model.rotating_frame_drive import RotatingFrameDrive
    from paraqeet.optimization_map import OptimizationMap
    from paraqeet.optimizers.dcrab_optimizer_gradient import DCRABOptimizerGradient
    from paraqeet.propagation.scipy_expm_grape import ScipyExpmGRAPE
    from paraqeet.quantity import Quantity
    from paraqeet.signal.envelopes import DCRABEnvelope
    from paraqeet.signal.pwc_generator import PWCGenerator
    from paraqeet.signal.waveform import FlatTopGaussianFilter


.. parsed-literal::

    datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).


Following the dCRAB optimization [Müller2022], we initialize our pulses
in the frequency domain by using the ``DCRABEnvelope``. Further we use a
flat-top Gaussian filter on top to ensure that the pulses start and end
at zero. And we use a ``PWCGenerator`` to generate the piece-wise
constant signal required for the ``GOAToverGRAPE`` method.

.. code:: ipython3

    # in seconds (See Eickbusch et al https://arxiv.org/abs/2111.06414 S9 A)
    delta_sampling = 33e-9
    n_pwc = 40  # number of piecewise constants in the pulse
    t_final = n_pwc * delta_sampling  # for now just set up for trying
    tlist = np.linspace(0, t_final, n_pwc + 1)
    eps_res = 3 * np.pi * 1.0  # initial amplitude of the resonator (in MHz)
    eps_max_res = 5 * eps_res  # maximum amplitude of the resonator (in MHz)
    eps_qubit = 3 * np.pi * 1.0  # initial amplitude of the qubit (in MHz)
    eps_max_qubit = 5 * eps_qubit  # maximum amplitude of the resonator (in MHz)
    
    tone_res = DCRABEnvelope(
        num_components=2,
        max_frequency=2 * np.pi * 2.0,
        amplitude=Quantity(eps_res * 1e6, -eps_max_res * 1e6, eps_max_res * 1e6, name="Amplitude CRAB resonator"),
        t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final, name="t_final CRAB resonator"),
        seed=46874,
    )
    
    smooth_tone_res = FlatTopGaussianFilter(
        envelopes=tone_res, t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final, name="t_final")
    )
    
    tone_qubit = DCRABEnvelope(
        num_components=2,
        max_frequency=2 * np.pi * 2.0,
        amplitude=Quantity(eps_qubit * 1e6, -eps_max_qubit * 1e6, eps_max_qubit * 1e6, name="Amplitude CRAB qubit"),
        t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final, name="t_final CRAB qubit"),
        seed=189841,
    )
    
    smooth_tone_qubit = FlatTopGaussianFilter(
        envelopes=tone_qubit, t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final, name="t_final")
    )
    
    gen_res = PWCGenerator(envelopes=[smooth_tone_res], tlist=tlist, max_amplitude=2 * eps_max_res * 1e6)
    gen_qubit = PWCGenerator(envelopes=[smooth_tone_qubit], tlist=tlist, max_amplitude=2 * eps_max_qubit * 1e6)

Here we plot the resonator tone and compare it to its smooth version.
Similarly one can plot the qubit tone.

.. code:: ipython3

    from plotting import plot_signal
    
    ts = np.linspace(0, t_final, 1001)
    fig, ax = plt.subplots(1, figsize=(5, 3))
    
    plot_signal(smooth_tone_res, ts, ax, linestyle="-", label="Smooth")
    plot_signal(gen_res, ts, ax, linestyle="--", label="PWC")
    
    plt.legend()
    plt.show()



.. image:: 08C_Bosonic_grape_with_smooth_pulses_files/08C_Bosonic_grape_with_smooth_pulses_5_0.png


Like the previous example, we define qubit and resonator parameters, as
well as the list of different Fock truncation numbers. We also set the
target Fock state.

For now we set the Fock-state truncation at small values (3 and 4) for
testing the method. Later we demonstrate the method with (30 and 31)
levels in the resonator.

.. code:: ipython3

    # Increasing the number of Fock states gives a more realistic scenario, at the price of increasing the simulation time
    n_fock_truncation_list = [3, 4]
    fock_target = 2
    n_times = 1001
    
    omega_range_factor = 1e-3  # just for initialization of the Quantity object
    omega_res = 2 * np.pi * 5.26e9
    omega_qubit = 2 * np.pi * 6.65e9
    
    omega_drive_res = omega_res
    omega_drive_qubit = omega_qubit
    
    detuning_drive_res = omega_res - omega_drive_res
    detuning_drive_qubit = omega_qubit - omega_drive_qubit
    
    drive_res = RotatingFrameDrive(gen_res)
    drive_qubit = RotatingFrameDrive(gen_qubit)

Similar to the previous example, following [Heeres2017], we thus
consider
:math:`N_{\mathrm{T}} \in \{N_{\mathrm{T}}^{(\mathrm{min})}, N_{\mathrm{T}}^{(\mathrm{min})} + 1, \dots, N_{\mathrm{T}}^{(\mathrm{max})} \}`,
and introduce a penalty when having different values of fidelities for
different truncation numbers. Thus, we create different systems, and
accordingly fidelity measures, for the different Fock truncation
numbers.

.. code:: ipython3

    def create_experiment(n_fock_truncation_list, fock_target):
        """
        Create the initial and target states, for the given Fock numbers.
        Also create the propagations and measurements for the given truncation numbers and target Fock states.
        """
        resonator_list = []
        coupling_list = []
        model_list = []
        hamiltonian_list = []
        prop_list = []
        initial_state_list = []
        target_state_list = []
        fock_number_op_list = []
        dcrab_fid_list = []
    
        qubit = Qubit(
            frequency=Quantity(
                value=detuning_drive_qubit,
                min_value=-omega_range_factor * omega_qubit,
                max_value=omega_range_factor * omega_qubit,
            ),
            drives=[drive_qubit],
        )
        ground_state_qubit = np.array([[0.0], [1.0]])
    
        # We take it to be 10 times the value in Eickbusch et al to speed up the simulation
        chi = 2 * np.pi * 32.81e3 * 10
    
        for n in n_fock_truncation_list:
            resonator = Resonator(
                frequency=Quantity(
                    value=detuning_drive_res,
                    min_value=-omega_range_factor * omega_res,
                    max_value=omega_range_factor * omega_res,
                ),
                dimension=n,
                drives=[drive_res],
            )
            resonator_list.append(resonator)
            coupling = TwoBodyCoupling(
                resonator,
                qubit,
                is_longitudinal=True,
                coefficient=Quantity(value=chi, min_value=chi / 2, max_value=2 * chi),
            )
            coupling_list.append(coupling)
            ham = CompositeHamiltonian([resonator, qubit], [coupling])
            hamiltonian_list.append(ham)
            model = ClosedSystem(hamiltonian=ham)
            model_list.append(model)
    
            # Propagation dt has to be less than `delta_sampling = 33e-9`. Set dt = 30e-9.
            prop = ScipyExpmGRAPE(model, resolution=1 / (30e-9))
    
            initial_res_state = np.zeros([n, 1], dtype=complex)
            initial_res_state[0, 0] = 1  # vacuum state
            initial_state = np.kron(initial_res_state, ground_state_qubit)
            initial_state_list.append(initial_state)
    
            target_res_state = np.zeros([n, 1], dtype=complex)
            target_res_state[fock_target, 0] = 1
            target_state = np.kron(target_res_state, ground_state_qubit)
            target_state_list.append(target_state)
    
            n_op = np.kron(np.diag([x for x in range(n)]), np.identity(2))
            fock_number_op_list.append(n_op)
    
            prop.set_initial_state(initial_state)
            prop.set_target_state(target_state)
            prop.use_schirmer_derivative = True
    
            prop_list.append(prop)
    
            fid = StateTransferFidelityGRAPE(propagation=prop, initial_state=initial_state, target_state=target_state)
            goat_over_grape_fid = GOATOverGRAPE(fid, prop, generators=[gen_res, gen_qubit])
            dcrab_fid_list.append(goat_over_grape_fid)
    
        return initial_state_list, target_state_list, fock_number_op_list, prop_list, dcrab_fid_list
    
    
    initial_state_list, target_state_list, fock_number_op_list, prop_list, dcrab_meas_list = create_experiment(
        n_fock_truncation_list, fock_target
    )

We consider as cost function of the form

.. math::


   C(\varepsilon(t) ) = w_1 \sum_{N = N_{\mathrm{T}}^{(\mathrm{min})}}^{ N_{\mathrm{T}}^{(\mathrm{max})}} \mathcal{F}_{N} (\varepsilon(t) )   - \frac{w_2}{2} \sum_{N, N' = N_{\mathrm{T}}^{(\mathrm{min})}}^{ N_{\mathrm{T}}^{(\mathrm{max})}} \left[\mathcal{F}_{N}(\varepsilon(t))- \mathcal{F}_{N'} (\varepsilon(t) ) \right]^2,

that we want to maximize. This cost weighted cost function can be
constructed using the class WeightedSumGoal.

*Note - As we consider smooth pulses as our basis function, we do not
need to add additional smoothness cost to the goal function.*

.. code:: ipython3

    weights = np.array([1.0 for _ in range(len(dcrab_meas_list))])
    weight_sum_of_squares = -0.05
    total_sum_of_weights = np.sum(weights) + weight_sum_of_squares
    weights = weights / total_sum_of_weights
    weight_sum_of_squares = weight_sum_of_squares / total_sum_of_weights
    dcrab_meas_list_bool = [True for meas in dcrab_meas_list]
    sum_of_squares_options = {"weight": weight_sum_of_squares, "meas_bool": dcrab_meas_list_bool}
    
    dcrab_goal = WeightedSumGoal(dcrab_meas_list, weights, sum_of_squares_options=sum_of_squares_options)

We can plot the initial pulses, as well as the corresponding relevant
dynamics.

.. code:: ipython3

    def plot_states_and_fock_number(item=0):
        """Plot the states."""
        ts = np.linspace(0, t_final, n_times)
        states = prop_list[item].propagate(ts)
        sig_res = gen_res.get_value(ts)
        sig_qubit = gen_qubit.get_value(ts)
        pop_initial_state = (np.abs(initial_state_list[item].conj().T @ states) ** 2).flatten()
        pop_target_state = (np.abs(target_state_list[item].conj().T @ states) ** 2).flatten()
    
        pop_initial_state = (np.abs(initial_state_list[item].conj().T @ states) ** 2).flatten()
    
        n_fock_avg = np.zeros(n_times, dtype=float)
        for k in range(n_times):
            n_fock_avg[k] = np.real((states[k].conj().T @ fock_number_op_list[item] @ states[k])[0, 0])
    
        _, ax = plt.subplots(4, figsize=(4, 10), sharex=True)
        ax[0].plot(ts / 1e-9, np.real(sig_res) / 1e6 / (2 * np.pi), label="I")
        ax[0].plot(ts / 1e-9, np.imag(sig_res) / 1e6 / (2 * np.pi), label="Q")
        ax[0].legend(loc=1)
        ax[0].set_ylabel(r"Field [MHz / $2\pi$]")
        ax[0].set_title("Resonator")
        ax[0].grid(True, linestyle=(1, (1, 5)), linewidth=1)
    
        ax[1].plot(ts / 1e-9, np.real(sig_qubit) / 1e6 / (2 * np.pi), label="I")
        ax[1].plot(ts / 1e-9, np.imag(sig_qubit) / 1e6 / (2 * np.pi), label="Q")
        ax[1].legend(loc=1)
        ax[1].set_ylabel(r"Field [MHz / $2\pi$]")
        ax[1].set_title("Qubit")
        ax[1].grid(True, linestyle=(1, (1, 5)), linewidth=1)
    
        ax[2].plot(ts / 1e-9, pop_initial_state, label="$|0,0 \\rangle$")
        ax[2].plot(ts / 1e-9, pop_target_state, label="$|2, 0 \\rangle$")
        ax[2].legend(loc="best")
        ax[2].set_ylabel("Population")
        ax[2].grid(True, linestyle=(1, (1, 5)), linewidth=1)
    
        ax[3].plot(ts / 1e-9, n_fock_avg)
        ax[3].set_ylabel("$\\langle n \\rangle$")
        if n_fock_truncation_list[item] < 10:
            y_fock_ticks = np.arange(n_fock_truncation_list[item])
        else:
            y_fock_ticks = np.arange(n_fock_truncation_list[item])[::5]
        ax[3].set_yticks(y_fock_ticks)
        ax[3].grid(axis="y", which="major")
        ax[3].minorticks_on()
        ax[3].grid(True, axis="x", linestyle=(1, (1, 5)), linewidth=1)
    
        ax[-1].set_xlabel("Time [ns]")
        plt.show()
    
    
    plot_states_and_fock_number()



.. image:: 08C_Bosonic_grape_with_smooth_pulses_files/08C_Bosonic_grape_with_smooth_pulses_13_0.png


And we compute the fidelities for the different truncation numbers

.. code:: ipython3

    print(f"Fidelity at N_T={n_fock_truncation_list[0]} = {dcrab_meas_list[0].measure(tlist)}")
    print(f"Fidelity at N_T={n_fock_truncation_list[1]} = {dcrab_meas_list[1].measure(tlist)}")


.. parsed-literal::

    Fidelity at N_T=3 = 0.09507140546701529


.. parsed-literal::

    Fidelity at N_T=4 = 0.040546120929160566


which are quite poor! We now proceed with the pulse optimization.

Lets define the parameters we want to optimize. Since, in this case we
want to optimize the smooth pulses, the parameters for the optmap would
be the parameters of the ``DCRABEnvelope`` tone.

.. code:: ipython3

    params_tone_qubit = tone_qubit.get_parameters()
    params_tone_res = tone_res.get_parameters()
    params_tone_qubit




.. parsed-literal::

    [Amplitude CRAB qubit: 9.42e+06,
     t_final CRAB qubit: 1.32e-06,
     CRAB Re coefficient 0: 0.742,
     CRAB Re coefficient 1: -0.875,
     CRAB Re frequency 0: 1.8 Hz x 2pi,
     CRAB Re frequency 1: 522 mHz x 2pi,
     CRAB Re Phase 0: 342 mHz x 2pi,
     CRAB Re Phase 1: 307 mHz x 2pi,
     CRAB Im coefficient 0: 0.632,
     CRAB Im coefficient 1: -0.163,
     CRAB Im frequency 0: 1.68 Hz x 2pi,
     CRAB Im frequency 1: 331 mHz x 2pi,
     CRAB Im Phase 0: 199 mHz x 2pi,
     CRAB Im Phase 1: -103 mHz x 2pi]



Further, we can write the optimization progress into a log file, which we
can later use to plot the infidelity vs function evaluation.

.. code:: ipython3

    import tempfile
    
    temp_dir = tempfile.TemporaryDirectory()
    print(f"Logging directory = {temp_dir.name}")
    
    max_iter = 1000  # set to 1000 for a good result; set to 10 for a quick example
    optmap = OptimizationMap()
    optmap.add(tone_res)
    optmap.add(tone_qubit)
    
    # Remove the t_final and  DRAG Delta from optimization from drag tone resonator and qubit
    optmap.remove(tone_qubit, params_tone_qubit[1])
    optmap.remove(tone_res, params_tone_res[1])
    optmap.register_params_with_optimizables()
    
    file_logger = FileLogger(temp_dir.name)
    
    # Define the optimizer
    opt = DCRABOptimizerGradient(
        dcrab_goal,
        optimization_map=optmap,
        super_iteration_every=100,
        max_super_iteration_num=2,
        print_every_iteration_num=20,
        super_iteration_tol=1e-6,
        seed=5648,
    )
    opt.set_options({"maxfun": max_iter, "workers": 32})
    opt.logger = file_logger


.. parsed-literal::

    Logging directory = /tmp/tmps582_4rw


.. code:: ipython3

    optmap




.. parsed-literal::

    ==== <class 'paraqeet.signal.envelopes.DCRABEnvelope'> ====
    [Amplitude CRAB resonator: 9.42e+06, CRAB Re coefficient 0: 0.807, CRAB Re coefficient 1: -0.529, CRAB Re frequency 0: 1.51 Hz x 2pi, CRAB Re frequency 1: 916 mHz x 2pi, CRAB Re Phase 0: 124 mHz x 2pi, CRAB Re Phase 1: -339 mHz x 2pi, CRAB Im coefficient 0: -0.614, CRAB Im coefficient 1: 0.803, CRAB Im frequency 0: 1.92 Hz x 2pi, CRAB Im frequency 1: 1.47 Hz x 2pi, CRAB Im Phase 0: -44 mHz x 2pi, CRAB Im Phase 1: 297 mHz x 2pi]
    
    ==== <class 'paraqeet.signal.envelopes.DCRABEnvelope'> ====
    [Amplitude CRAB qubit: 9.42e+06, CRAB Re coefficient 0: 0.742, CRAB Re coefficient 1: -0.875, CRAB Re frequency 0: 1.8 Hz x 2pi, CRAB Re frequency 1: 522 mHz x 2pi, CRAB Re Phase 0: 342 mHz x 2pi, CRAB Re Phase 1: 307 mHz x 2pi, CRAB Im coefficient 0: 0.632, CRAB Im coefficient 1: -0.163, CRAB Im frequency 0: 1.68 Hz x 2pi, CRAB Im frequency 1: 331 mHz x 2pi, CRAB Im Phase 0: 199 mHz x 2pi, CRAB Im Phase 1: -103 mHz x 2pi]




.. code:: ipython3

    opt.optimize(gen_res.tlist)


.. parsed-literal::

    scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.


.. parsed-literal::

    Iteration number = 0 	  Infidelity  = 2.971e-01


.. parsed-literal::

    Iteration number = 20 	  Infidelity  = 2.632e-02


.. parsed-literal::

    Iteration number = 40 	  Infidelity  = 2.198e-02


.. parsed-literal::

    Iteration number = 60 	  Infidelity  = 1.950e-02


.. parsed-literal::

    Iteration number = 80 	  Infidelity  = 1.866e-02


.. parsed-literal::

    
    
    ==== Max iteration before a super-iteration reached ====
    ==== Starting super-iteration 1 ====
    *** Current lowest infidelity =  0.019 ***
    * Current no. of parameters = 50


.. parsed-literal::

    Iteration number = 100 	  Infidelity  = 2.514e-01


.. parsed-literal::

    Iteration number = 120 	  Infidelity  = 4.825e-02


.. parsed-literal::

    Iteration number = 140 	  Infidelity  = 3.104e-02


.. parsed-literal::

    Iteration number = 160 	  Infidelity  = 1.664e-02


.. parsed-literal::

    Iteration number = 180 	  Infidelity  = 1.371e-02


.. parsed-literal::

    
    
    ==== Max iteration before a super-iteration reached ====
    ==== Starting super-iteration 2 ====
    *** Current lowest infidelity =  0.012 ***
    * Current no. of parameters = 74


.. parsed-literal::

    Iteration number = 200 	  Infidelity  = 6.805e-01


.. parsed-literal::

    Stopping the optimization or backtracking to previous best fidelity. Going to step with 50 parameters.


.. parsed-literal::

    Stopping the optimization or backtracking to previous best fidelity. Going to step with 26 parameters.


.. parsed-literal::

    Setting parameters to the best values.




.. parsed-literal::

    {'status': 2, 'value': 0.012016004582277917, 'iterations': 127, 'message': '`callback` raised `StopIteration`.'}



Lets set the optimization to the best parameters obtained during the run

.. code:: ipython3

    opt.set_parameters(opt.best_params)




.. parsed-literal::

    [Amplitude CRAB resonator: 4.26e+07,
     CRAB Re coefficient 0: 0.688,
     CRAB Re coefficient 1: -0.776,
     CRAB Re coefficient 2: 0.337,
     CRAB Re coefficient 3: 0.282,
     CRAB Re coefficient 4: 0,
     CRAB Re coefficient 5: 0,
     CRAB Re frequency 0: 1.45 Hz x 2pi,
     CRAB Re frequency 1: 767 mHz x 2pi,
     CRAB Re frequency 2: 808 mHz x 2pi,
     CRAB Re frequency 3: 1.55 Hz x 2pi,
     CRAB Re frequency 4: 1 Hz x 2pi,
     CRAB Re frequency 5: 1 Hz x 2pi,
     CRAB Re Phase 0: 74.9 mHz x 2pi,
     CRAB Re Phase 1: -348 mHz x 2pi,
     CRAB Re Phase 2: 436 mHz x 2pi,
     CRAB Re Phase 3: -255 mHz x 2pi,
     CRAB Re Phase 4: 0 Hz x 2pi,
     CRAB Re Phase 5: 0 Hz x 2pi,
     CRAB Im coefficient 0: -0.394,
     CRAB Im coefficient 1: 0.967,
     CRAB Im coefficient 2: 0.126,
     CRAB Im coefficient 3: -0.000265,
     CRAB Im coefficient 4: 0,
     CRAB Im coefficient 5: 0,
     CRAB Im frequency 0: 1.65 Hz x 2pi,
     CRAB Im frequency 1: 1.86 Hz x 2pi,
     CRAB Im frequency 2: 323 mHz x 2pi,
     CRAB Im frequency 3: 1.14 Hz x 2pi,
     CRAB Im frequency 4: 1 Hz x 2pi,
     CRAB Im frequency 5: 1 Hz x 2pi,
     CRAB Im Phase 0: -28.7 mHz x 2pi,
     CRAB Im Phase 1: 480 mHz x 2pi,
     CRAB Im phase 2: -304 mHz x 2pi,
     CRAB Im phase 3: -380 mHz x 2pi,
     CRAB Im phase 4: 0 Hz x 2pi,
     CRAB Im phase 5: 0 Hz x 2pi,
     Amplitude CRAB qubit: -4.61e+07,
     CRAB Re coefficient 0: 0.784,
     CRAB Re coefficient 1: -0.843,
     CRAB Re coefficient 2: -0.0338,
     CRAB Re coefficient 3: 0.343,
     CRAB Re coefficient 4: 0,
     CRAB Re coefficient 5: 0,
     CRAB Re frequency 0: 1.7 Hz x 2pi,
     CRAB Re frequency 1: 1.02 Hz x 2pi,
     CRAB Re frequency 2: 1.36 Hz x 2pi,
     CRAB Re frequency 3: 859 mHz x 2pi,
     CRAB Re frequency 4: 1 Hz x 2pi,
     CRAB Re frequency 5: 1 Hz x 2pi,
     CRAB Re Phase 0: 418 mHz x 2pi,
     CRAB Re Phase 1: 491 mHz x 2pi,
     CRAB Re Phase 2: -326 mHz x 2pi,
     CRAB Re Phase 3: -182 mHz x 2pi,
     CRAB Re Phase 4: 0 Hz x 2pi,
     CRAB Re Phase 5: 0 Hz x 2pi,
     CRAB Im coefficient 0: 0.56,
     CRAB Im coefficient 1: -0.373,
     CRAB Im coefficient 2: 0.0206,
     CRAB Im coefficient 3: 0.00603,
     CRAB Im coefficient 4: 0,
     CRAB Im coefficient 5: 0,
     CRAB Im frequency 0: 1.89 Hz x 2pi,
     CRAB Im frequency 1: 645 mHz x 2pi,
     CRAB Im frequency 2: 981 mHz x 2pi,
     CRAB Im frequency 3: 1.43 Hz x 2pi,
     CRAB Im frequency 4: 1 Hz x 2pi,
     CRAB Im frequency 5: 1 Hz x 2pi,
     CRAB Im Phase 0: 158 mHz x 2pi,
     CRAB Im Phase 1: -60.3 mHz x 2pi,
     CRAB Im phase 2: -296 mHz x 2pi,
     CRAB Im phase 3: 19.6 mHz x 2pi,
     CRAB Im phase 4: 0 Hz x 2pi,
     CRAB Im phase 5: 0 Hz x 2pi]



We can now plot the new pulses and the corresponding dynamics. If the
number of iterations (max_iter) is chosen to be large enough, we should
see an improvement.

.. code:: ipython3

    plot_states_and_fock_number()



.. image:: 08C_Bosonic_grape_with_smooth_pulses_files/08C_Bosonic_grape_with_smooth_pulses_26_0.png


We can also compare the dynamics with the higher truncation number.
Ideally, if the pulse amplitude is not high enough to produce artifacts
due to the truncation of Hilbert space, the dynamics in the two cases
should match each other. Else one needs to either reduce the pulse
amplitude or increase the truncation cutoff.

.. code:: ipython3

    plot_states_and_fock_number(1)



.. image:: 08C_Bosonic_grape_with_smooth_pulses_files/08C_Bosonic_grape_with_smooth_pulses_28_0.png


The new fidelities are

.. code:: ipython3

    print(f"Fidelity at N_T={n_fock_truncation_list[0]} = {dcrab_meas_list[0].measure(tlist)}")
    print(f"Fidelity at N_T={n_fock_truncation_list[1]} = {dcrab_meas_list[1].measure(tlist)}")


.. parsed-literal::

    Fidelity at N_T=3 = 0.9755915686011631
    Fidelity at N_T=4 = 0.9510002321911987


For this small truncation number the dynamics and fidelities do not
match very well. This indicates the need for higher truncation numbers,
which we demonstrate in the next section.

Finally, lets plot the variation of infidelity with evaluation number.

.. code:: ipython3

    from plotting import plot_infidelity_vs_evaluation_from_logs
    
    plot_infidelity_vs_evaluation_from_logs(log_path=temp_dir.name + "/opt.log", label="dCRAB optimization");



.. image:: 08C_Bosonic_grape_with_smooth_pulses_files/08C_Bosonic_grape_with_smooth_pulses_33_0.png


Setting truncation to higher value
----------------------------------

Here we set the truncation values to 30 and 31 levels for a more
realistic simulation and rerun the entire simulation.

*Note - The following takes about 60 mins to run on an AMD-EPYC Milan
processor with 128 cores and 256GB RAM (might take more depending on the
number of CPU cores available).*

We first reset the generator parameters (by redefining them), and
redefine the resonator with higher truncation numbers.

.. code:: ipython3

    # in seconds (See Eickbusch et al https://arxiv.org/abs/2111.06414 S9 A)
    delta_sampling = 33e-9
    n_pwc = 40  # number of piecewise constants in the pulse
    t_final = n_pwc * delta_sampling  # for now just set up for trying
    tlist = np.linspace(0, t_final, n_pwc + 1)
    eps_res = 3 * np.pi * 1.0  # initial amplitude of the resonator (in MHz)
    eps_max_res = 5 * eps_res  # maximum amplitude of the resonator (in MHz)
    eps_qubit = 3 * np.pi * 1.0  # initial amplitude of the qubit (in MHz)
    eps_max_qubit = 5 * eps_qubit  # maximum amplitude of the resonator (in MHz)
    
    tone_res = DCRABEnvelope(
        num_components=3,
        max_frequency=2 * np.pi * 5.0,
        amplitude=Quantity(eps_res * 1e6, -eps_max_res * 1e6, eps_max_res * 1e6, name="Amplitude CRAB resonator"),
        t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final, name="t_final CRAB resonator"),
        seed=17619276394480986,
    )
    
    smooth_tone_res = FlatTopGaussianFilter(
        envelopes=tone_res, t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final, name="t_final")
    )
    
    tone_qubit = DCRABEnvelope(
        num_components=3,
        max_frequency=2 * np.pi * 5.0,
        amplitude=Quantity(eps_qubit * 1e6, -eps_max_qubit * 1e6, eps_max_qubit * 1e6, name="Amplitude CRAB qubit"),
        t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final, name="t_final CRAB qubit"),
        seed=17619276394849436,
    )
    
    smooth_tone_qubit = FlatTopGaussianFilter(
        envelopes=tone_qubit, t_final=Quantity(t_final, 1 / 2 * t_final, 2 * t_final, name="t_final")
    )
    
    gen_res = PWCGenerator(envelopes=[smooth_tone_res], tlist=tlist, max_amplitude=2 * eps_max_res * 1e6)
    gen_qubit = PWCGenerator(envelopes=[smooth_tone_qubit], tlist=tlist, max_amplitude=2 * eps_max_qubit * 71e6)

.. code:: ipython3

    # Increasing the number of Fock states gives a more realistic scenario, at the price of increasing the simulation time
    n_fock_truncation_list = [30, 31]
    fock_target = 2
    n_times = 1001
    
    omega_range_factor = 1e-3  # just for initialization of the Quantity object
    omega_res = 2 * np.pi * 5.26e9
    omega_qubit = 2 * np.pi * 6.65e9
    
    omega_drive_res = omega_res
    omega_drive_qubit = omega_qubit
    
    detuning_drive_res = omega_res - omega_drive_res
    detuning_drive_qubit = omega_qubit - omega_drive_qubit
    
    drive_res = RotatingFrameDrive(gen_res)
    drive_qubit = RotatingFrameDrive(gen_qubit)
    
    
    # Create a new experiment and redefine the measurement
    initial_state_list, target_state_list, fock_number_op_list, prop_list, dcrab_meas_list = create_experiment(
        n_fock_truncation_list, fock_target
    )
    
    weights = np.array([1.0 for _ in range(len(dcrab_meas_list))])
    weight_sum_of_squares = -0.05
    total_sum_of_weights = np.sum(weights) + weight_sum_of_squares
    weights = weights / total_sum_of_weights
    weight_sum_of_squares = weight_sum_of_squares / total_sum_of_weights
    dcrab_meas_list_bool = [True for meas in dcrab_meas_list]
    sum_of_squares_options = {"weight": weight_sum_of_squares, "meas_bool": dcrab_meas_list_bool}
    
    dcrab_goal = WeightedSumGoal(dcrab_meas_list, weights, sum_of_squares_options=sum_of_squares_options)

.. code:: ipython3

    plot_states_and_fock_number()



.. image:: 08C_Bosonic_grape_with_smooth_pulses_files/08C_Bosonic_grape_with_smooth_pulses_37_0.png


.. code:: ipython3

    plot_states_and_fock_number(1)



.. image:: 08C_Bosonic_grape_with_smooth_pulses_files/08C_Bosonic_grape_with_smooth_pulses_38_0.png


Initial fidelity before optimization

.. code:: ipython3

    print(f"Fidelity at N_T={n_fock_truncation_list[0]} = {dcrab_meas_list[0].measure(tlist)}")
    print(f"Fidelity at N_T={n_fock_truncation_list[1]} = {dcrab_meas_list[1].measure(tlist)}")


.. parsed-literal::

    Fidelity at N_T=30 = 0.00809359027771987


.. parsed-literal::

    Fidelity at N_T=31 = 0.00809359027771987


.. code:: ipython3

    tone_res.get_parameters()




.. parsed-literal::

    [Amplitude CRAB resonator: 9.42e+06,
     t_final CRAB resonator: 1.32e-06,
     CRAB Re coefficient 0: -0.218,
     CRAB Re coefficient 1: -0.711,
     CRAB Re coefficient 2: -0.223,
     CRAB Re frequency 0: 4.77 Hz x 2pi,
     CRAB Re frequency 1: 4.34 Hz x 2pi,
     CRAB Re frequency 2: 4.59 Hz x 2pi,
     CRAB Re Phase 0: 478 mHz x 2pi,
     CRAB Re Phase 1: 5.23 mHz x 2pi,
     CRAB Re Phase 2: 415 mHz x 2pi,
     CRAB Im coefficient 0: -0.725,
     CRAB Im coefficient 1: -0.262,
     CRAB Im coefficient 2: 0.901,
     CRAB Im frequency 0: 3.33 Hz x 2pi,
     CRAB Im frequency 1: 3.18 Hz x 2pi,
     CRAB Im frequency 2: 1.97 Hz x 2pi,
     CRAB Im Phase 0: 205 mHz x 2pi,
     CRAB Im Phase 1: 230 mHz x 2pi,
     CRAB Im Phase 2: 60.2 mHz x 2pi]



.. code:: ipython3

    tone_qubit.get_parameters()




.. parsed-literal::

    [Amplitude CRAB qubit: 9.42e+06,
     t_final CRAB qubit: 1.32e-06,
     CRAB Re coefficient 0: -0.693,
     CRAB Re coefficient 1: 0.833,
     CRAB Re coefficient 2: 0.687,
     CRAB Re frequency 0: 4.03 Hz x 2pi,
     CRAB Re frequency 1: 2.46 Hz x 2pi,
     CRAB Re frequency 2: 4.43 Hz x 2pi,
     CRAB Re Phase 0: -98.2 mHz x 2pi,
     CRAB Re Phase 1: -341 mHz x 2pi,
     CRAB Re Phase 2: -211 mHz x 2pi,
     CRAB Im coefficient 0: -0.152,
     CRAB Im coefficient 1: -0.103,
     CRAB Im coefficient 2: 0.579,
     CRAB Im frequency 0: 2.09 Hz x 2pi,
     CRAB Im frequency 1: 4.5 Hz x 2pi,
     CRAB Im frequency 2: 2.08 Hz x 2pi,
     CRAB Im Phase 0: -353 mHz x 2pi,
     CRAB Im Phase 1: 481 mHz x 2pi,
     CRAB Im Phase 2: -16.3 mHz x 2pi]



Redefine the optmap and the optimizer and rerun the optimization

.. code:: ipython3

    import tempfile
    
    temp_dir = tempfile.TemporaryDirectory()
    print(f"Logging directory = {temp_dir.name}")
    
    max_iter = 1000  # set to 1000 for a good result; set to 10 for a quick example
    optmap = OptimizationMap()
    optmap.add(tone_res)
    optmap.add(tone_qubit)
    
    # Remove the t_final and  DRAG Delta from optimization from drag tone resonator and qubit
    optmap.remove(tone_qubit, params_tone_qubit[1])
    optmap.remove(tone_res, params_tone_res[1])
    optmap.register_params_with_optimizables()
    
    file_logger = FileLogger(temp_dir.name)
    
    # Define the optimizer
    opt = DCRABOptimizerGradient(
        dcrab_goal,
        optimization_map=optmap,
        super_iteration_every=500,
        max_super_iteration_num=3,
        print_every_iteration_num=10,
        super_iteration_tol=1e-6,
        seed=8647,
    )
    opt.set_options({"maxfun": max_iter, "workers": 32})
    opt.logger = file_logger


.. parsed-literal::

    Logging directory = /tmp/tmp21kfkspm


.. parsed-literal::

    Implicitly cleaning up <TemporaryDirectory '/tmp/tmps582_4rw'>


.. code:: ipython3

    opt.optimize(gen_res.tlist)
    opt.set_parameters(opt.best_params)


.. parsed-literal::

    Iteration number = 0 	  Infidelity  = 9.039e-01


.. parsed-literal::

    Iteration number = 10 	  Infidelity  = 7.075e-01


.. parsed-literal::

    Iteration number = 20 	  Infidelity  = 6.510e-01


.. parsed-literal::

    Iteration number = 30 	  Infidelity  = 5.641e-01


.. parsed-literal::

    Iteration number = 40 	  Infidelity  = 5.279e-01


.. parsed-literal::

    Iteration number = 50 	  Infidelity  = 4.911e-01


.. parsed-literal::

    Iteration number = 60 	  Infidelity  = 4.726e-01


.. parsed-literal::

    Iteration number = 70 	  Infidelity  = 4.560e-01


.. parsed-literal::

    Iteration number = 80 	  Infidelity  = 4.453e-01


.. parsed-literal::

    Iteration number = 90 	  Infidelity  = 4.328e-01


.. parsed-literal::

    
    
    ==== Decrease in infidelity less than 1e-06 ====
    ==== Starting super-iteration 1 ====
    * Current lowest infidelity =  4.279e-01
    * Current no. of parameters = 74


.. parsed-literal::

    Iteration number = 100 	  Infidelity  = 8.588e-01


.. parsed-literal::

    Iteration number = 110 	  Infidelity  = 7.227e-01


.. parsed-literal::

    Iteration number = 120 	  Infidelity  = 7.221e-01


.. parsed-literal::

    Iteration number = 130 	  Infidelity  = 7.191e-01


.. parsed-literal::

    Iteration number = 140 	  Infidelity  = 6.993e-01


.. parsed-literal::

    Iteration number = 150 	  Infidelity  = 6.798e-01


.. parsed-literal::

    Iteration number = 160 	  Infidelity  = 6.485e-01


.. parsed-literal::

    Iteration number = 170 	  Infidelity  = 6.276e-01


.. parsed-literal::

    Iteration number = 180 	  Infidelity  = 6.140e-01


.. parsed-literal::

    Iteration number = 190 	  Infidelity  = 5.947e-01


.. parsed-literal::

    Iteration number = 200 	  Infidelity  = 5.675e-01


.. parsed-literal::

    Iteration number = 210 	  Infidelity  = 5.487e-01


.. parsed-literal::

    Iteration number = 220 	  Infidelity  = 5.158e-01


.. parsed-literal::

    Iteration number = 230 	  Infidelity  = 4.893e-01


.. parsed-literal::

    Iteration number = 240 	  Infidelity  = 4.578e-01


.. parsed-literal::

    Iteration number = 250 	  Infidelity  = 4.325e-01


.. parsed-literal::

    Iteration number = 260 	  Infidelity  = 4.167e-01


.. parsed-literal::

    Iteration number = 270 	  Infidelity  = 4.061e-01


.. parsed-literal::

    Iteration number = 280 	  Infidelity  = 3.964e-01


.. parsed-literal::

    Iteration number = 290 	  Infidelity  = 3.772e-01


.. parsed-literal::

    Iteration number = 300 	  Infidelity  = 3.657e-01


.. parsed-literal::

    Iteration number = 310 	  Infidelity  = 3.508e-01


.. parsed-literal::

    Iteration number = 320 	  Infidelity  = 3.298e-01


.. parsed-literal::

    Iteration number = 330 	  Infidelity  = 3.243e-01


.. parsed-literal::

    Iteration number = 340 	  Infidelity  = 3.180e-01


.. parsed-literal::

    Iteration number = 350 	  Infidelity  = 3.122e-01


.. parsed-literal::

    Iteration number = 360 	  Infidelity  = 3.093e-01


.. parsed-literal::

    Iteration number = 370 	  Infidelity  = 3.064e-01


.. parsed-literal::

    Iteration number = 380 	  Infidelity  = 3.053e-01


.. parsed-literal::

    Iteration number = 390 	  Infidelity  = 2.997e-01


.. parsed-literal::

    Iteration number = 400 	  Infidelity  = 2.915e-01


.. parsed-literal::

    Iteration number = 410 	  Infidelity  = 2.874e-01


.. parsed-literal::

    Iteration number = 420 	  Infidelity  = 2.844e-01


.. parsed-literal::

    Iteration number = 430 	  Infidelity  = 2.812e-01


.. parsed-literal::

    Iteration number = 440 	  Infidelity  = 2.773e-01


.. parsed-literal::

    Iteration number = 450 	  Infidelity  = 2.704e-01


.. parsed-literal::

    Iteration number = 460 	  Infidelity  = 2.650e-01


.. parsed-literal::

    Iteration number = 470 	  Infidelity  = 2.590e-01


.. parsed-literal::

    Iteration number = 480 	  Infidelity  = 2.533e-01


.. parsed-literal::

    
    
    ==== Decrease in infidelity less than 1e-06 ====
    ==== Starting super-iteration 2 ====
    * Current lowest infidelity =  2.504e-01
    * Current no. of parameters = 110


.. parsed-literal::

    Iteration number = 490 	  Infidelity  = 6.025e-01


.. parsed-literal::

    Iteration number = 500 	  Infidelity  = 3.114e-01


.. parsed-literal::

    Iteration number = 510 	  Infidelity  = 2.798e-01


.. parsed-literal::

    Iteration number = 520 	  Infidelity  = 2.611e-01


.. parsed-literal::

    Iteration number = 530 	  Infidelity  = 2.559e-01


.. parsed-literal::

    Iteration number = 540 	  Infidelity  = 2.448e-01


.. parsed-literal::

    Iteration number = 550 	  Infidelity  = 2.242e-01


.. parsed-literal::

    Iteration number = 560 	  Infidelity  = 2.123e-01


.. parsed-literal::

    Iteration number = 570 	  Infidelity  = 2.032e-01


.. parsed-literal::

    Iteration number = 580 	  Infidelity  = 1.987e-01


.. parsed-literal::

    Iteration number = 590 	  Infidelity  = 1.960e-01


.. parsed-literal::

    Iteration number = 600 	  Infidelity  = 1.940e-01


.. parsed-literal::

    Iteration number = 610 	  Infidelity  = 1.902e-01


.. parsed-literal::

    Iteration number = 620 	  Infidelity  = 1.855e-01


.. parsed-literal::

    Iteration number = 630 	  Infidelity  = 1.795e-01


.. parsed-literal::

    Iteration number = 640 	  Infidelity  = 1.750e-01


.. parsed-literal::

    Iteration number = 650 	  Infidelity  = 1.727e-01


.. parsed-literal::

    Iteration number = 660 	  Infidelity  = 1.693e-01


.. parsed-literal::

    Iteration number = 670 	  Infidelity  = 1.669e-01


.. parsed-literal::

    Iteration number = 680 	  Infidelity  = 1.628e-01


.. parsed-literal::

    Iteration number = 690 	  Infidelity  = 1.585e-01


.. parsed-literal::

    Iteration number = 700 	  Infidelity  = 1.556e-01


.. parsed-literal::

    Iteration number = 710 	  Infidelity  = 1.538e-01


.. parsed-literal::

    Iteration number = 720 	  Infidelity  = 1.503e-01


.. parsed-literal::

    Iteration number = 730 	  Infidelity  = 1.473e-01


.. parsed-literal::

    Iteration number = 740 	  Infidelity  = 1.435e-01


.. parsed-literal::

    Iteration number = 750 	  Infidelity  = 1.406e-01


.. parsed-literal::

    Iteration number = 760 	  Infidelity  = 1.396e-01


.. parsed-literal::

    Iteration number = 770 	  Infidelity  = 1.388e-01


.. parsed-literal::

    Iteration number = 780 	  Infidelity  = 1.369e-01


.. parsed-literal::

    
    
    ==== Decrease in infidelity less than 1e-06 ====
    ==== Starting super-iteration 3 ====
    * Current lowest infidelity =  1.362e-01
    * Current no. of parameters = 146


.. parsed-literal::

    Iteration number = 790 	  Infidelity  = 8.802e-01


.. parsed-literal::

    Stopping the optimization or backtracking to previous best fidelity. Going to step with 110 parameters.


.. parsed-literal::

    Stopping the optimization or backtracking to previous best fidelity. Going to step with 74 parameters.


.. parsed-literal::

    Stopping the optimization or backtracking to previous best fidelity. Going to step with 38 parameters.


.. parsed-literal::

    Setting parameters to the best values.




.. parsed-literal::

    [Amplitude CRAB resonator: 4.71e+07,
     CRAB Re coefficient 0: -0.994,
     CRAB Re coefficient 1: 0.0351,
     CRAB Re coefficient 2: -0.998,
     CRAB Re coefficient 3: 0.000167,
     CRAB Re coefficient 4: 0.00117,
     CRAB Re coefficient 5: 4.08e-05,
     CRAB Re coefficient 6: -0.0869,
     CRAB Re coefficient 7: -0.0453,
     CRAB Re coefficient 8: 0.000124,
     CRAB Re coefficient 9: 0,
     CRAB Re coefficient 10: 0,
     CRAB Re coefficient 11: 0,
     CRAB Re frequency 0: 3.68 Hz x 2pi,
     CRAB Re frequency 1: 48.9 mHz x 2pi,
     CRAB Re frequency 2: 3.69 Hz x 2pi,
     CRAB Re frequency 3: 4.93 Hz x 2pi,
     CRAB Re frequency 4: 4.58 Hz x 2pi,
     CRAB Re frequency 5: 4.68 Hz x 2pi,
     CRAB Re frequency 6: 3.66 Hz x 2pi,
     CRAB Re frequency 7: 1.12 Hz x 2pi,
     CRAB Re frequency 8: 3.02 Hz x 2pi,
     CRAB Re frequency 9: 2.5 Hz x 2pi,
     CRAB Re frequency 10: 2.5 Hz x 2pi,
     CRAB Re frequency 11: 2.5 Hz x 2pi,
     CRAB Re Phase 0: -176 mHz x 2pi,
     CRAB Re Phase 1: 464 mHz x 2pi,
     CRAB Re Phase 2: -177 mHz x 2pi,
     CRAB Re Phase 3: 303 mHz x 2pi,
     CRAB Re Phase 4: -448 mHz x 2pi,
     CRAB Re Phase 5: -500 mHz x 2pi,
     CRAB Re Phase 6: -159 mHz x 2pi,
     CRAB Re Phase 7: -189 mHz x 2pi,
     CRAB Re Phase 8: 382 mHz x 2pi,
     CRAB Re Phase 9: 0 Hz x 2pi,
     CRAB Re Phase 10: 0 Hz x 2pi,
     CRAB Re Phase 11: 0 Hz x 2pi,
     CRAB Im coefficient 0: -0.475,
     CRAB Im coefficient 1: -0.138,
     CRAB Im coefficient 2: 0.0181,
     CRAB Im coefficient 3: -0.00454,
     CRAB Im coefficient 4: -0.998,
     CRAB Im coefficient 5: 0.81,
     CRAB Im coefficient 6: -0.00109,
     CRAB Im coefficient 7: -0.0236,
     CRAB Im coefficient 8: 0.000695,
     CRAB Im coefficient 9: 0,
     CRAB Im coefficient 10: 0,
     CRAB Im coefficient 11: 0,
     CRAB Im frequency 0: 4.62 Hz x 2pi,
     CRAB Im frequency 1: 4.46 Hz x 2pi,
     CRAB Im frequency 2: 4.94 Hz x 2pi,
     CRAB Im frequency 3: 4.21 Hz x 2pi,
     CRAB Im frequency 4: 2.02 Hz x 2pi,
     CRAB Im frequency 5: 1.21 Hz x 2pi,
     CRAB Im frequency 6: 3.18 Hz x 2pi,
     CRAB Im frequency 7: 4.79 Hz x 2pi,
     CRAB Im frequency 8: 3.17 Hz x 2pi,
     CRAB Im frequency 9: 2.5 Hz x 2pi,
     CRAB Im frequency 10: 2.5 Hz x 2pi,
     CRAB Im frequency 11: 2.5 Hz x 2pi,
     CRAB Im Phase 0: 446 mHz x 2pi,
     CRAB Im Phase 1: 470 mHz x 2pi,
     CRAB Im Phase 2: -256 mHz x 2pi,
     CRAB Im phase 3: 249 mHz x 2pi,
     CRAB Im phase 4: -416 mHz x 2pi,
     CRAB Im phase 5: 412 mHz x 2pi,
     CRAB Im phase 6: 322 mHz x 2pi,
     CRAB Im phase 7: 123 mHz x 2pi,
     CRAB Im phase 8: -203 mHz x 2pi,
     CRAB Im phase 9: 0 Hz x 2pi,
     CRAB Im phase 10: 0 Hz x 2pi,
     CRAB Im phase 11: 0 Hz x 2pi,
     Amplitude CRAB qubit: 4.71e+07,
     CRAB Re coefficient 0: -0.868,
     CRAB Re coefficient 1: 0.0129,
     CRAB Re coefficient 2: -0.138,
     CRAB Re coefficient 3: -0.279,
     CRAB Re coefficient 4: -0.757,
     CRAB Re coefficient 5: -0.641,
     CRAB Re coefficient 6: -0.151,
     CRAB Re coefficient 7: 0.238,
     CRAB Re coefficient 8: -0.226,
     CRAB Re coefficient 9: 0,
     CRAB Re coefficient 10: 0,
     CRAB Re coefficient 11: 0,
     CRAB Re frequency 0: 4.27 Hz x 2pi,
     CRAB Re frequency 1: 4.53 Hz x 2pi,
     CRAB Re frequency 2: 4.08 Hz x 2pi,
     CRAB Re frequency 3: 3.66 Hz x 2pi,
     CRAB Re frequency 4: 1.95 Hz x 2pi,
     CRAB Re frequency 5: 3.51 Hz x 2pi,
     CRAB Re frequency 6: 1.28 Hz x 2pi,
     CRAB Re frequency 7: 1.12 Hz x 2pi,
     CRAB Re frequency 8: 772 mHz x 2pi,
     CRAB Re frequency 9: 2.5 Hz x 2pi,
     CRAB Re frequency 10: 2.5 Hz x 2pi,
     CRAB Re frequency 11: 2.5 Hz x 2pi,
     CRAB Re Phase 0: -409 mHz x 2pi,
     CRAB Re Phase 1: -275 mHz x 2pi,
     CRAB Re Phase 2: -448 mHz x 2pi,
     CRAB Re Phase 3: -418 mHz x 2pi,
     CRAB Re Phase 4: -335 mHz x 2pi,
     CRAB Re Phase 5: 450 mHz x 2pi,
     CRAB Re Phase 6: -318 mHz x 2pi,
     CRAB Re Phase 7: 405 mHz x 2pi,
     CRAB Re Phase 8: -151 mHz x 2pi,
     CRAB Re Phase 9: 0 Hz x 2pi,
     CRAB Re Phase 10: 0 Hz x 2pi,
     CRAB Re Phase 11: 0 Hz x 2pi,
     CRAB Im coefficient 0: -0.704,
     CRAB Im coefficient 1: 0.363,
     CRAB Im coefficient 2: 0.569,
     CRAB Im coefficient 3: 0.911,
     CRAB Im coefficient 4: -0.324,
     CRAB Im coefficient 5: 0.0387,
     CRAB Im coefficient 6: 0.521,
     CRAB Im coefficient 7: -0.381,
     CRAB Im coefficient 8: 0.0245,
     CRAB Im coefficient 9: 0,
     CRAB Im coefficient 10: 0,
     CRAB Im coefficient 11: 0,
     CRAB Im frequency 0: 5 Hz x 2pi,
     CRAB Im frequency 1: 4.39 Hz x 2pi,
     CRAB Im frequency 2: 4.57 Hz x 2pi,
     CRAB Im frequency 3: 4.17 Hz x 2pi,
     CRAB Im frequency 4: 4.62 Hz x 2pi,
     CRAB Im frequency 5: 4.41 Hz x 2pi,
     CRAB Im frequency 6: 4.03 Hz x 2pi,
     CRAB Im frequency 7: 3.16 Hz x 2pi,
     CRAB Im frequency 8: 3.91 Hz x 2pi,
     CRAB Im frequency 9: 2.5 Hz x 2pi,
     CRAB Im frequency 10: 2.5 Hz x 2pi,
     CRAB Im frequency 11: 2.5 Hz x 2pi,
     CRAB Im Phase 0: -426 mHz x 2pi,
     CRAB Im Phase 1: 301 mHz x 2pi,
     CRAB Im Phase 2: 89.4 mHz x 2pi,
     CRAB Im phase 3: -30.7 mHz x 2pi,
     CRAB Im phase 4: -192 mHz x 2pi,
     CRAB Im phase 5: -456 mHz x 2pi,
     CRAB Im phase 6: -60.1 mHz x 2pi,
     CRAB Im phase 7: 381 mHz x 2pi,
     CRAB Im phase 8: -349 mHz x 2pi,
     CRAB Im phase 9: 0 Hz x 2pi,
     CRAB Im phase 10: 0 Hz x 2pi,
     CRAB Im phase 11: 0 Hz x 2pi]



And the new fidelities are

.. code:: ipython3

    print(f"Fidelity at N_T={n_fock_truncation_list[0]} = {dcrab_meas_list[0].measure(tlist)}")
    print(f"Fidelity at N_T={n_fock_truncation_list[1]} = {dcrab_meas_list[1].measure(tlist)}")


.. parsed-literal::

    Fidelity at N_T=30 = 0.842357380859644


.. parsed-literal::

    Fidelity at N_T=31 = 0.8422578804548017


Here the dynamics and the fidelities are very similar to one another,
indicating no truncation artifacts.

Lets plot the optimized pulses and dynamics

.. code:: ipython3

    plot_states_and_fock_number()



.. image:: 08C_Bosonic_grape_with_smooth_pulses_files/08C_Bosonic_grape_with_smooth_pulses_50_0.png


And verify the dynamics with the higher truncation looks the same

.. code:: ipython3

    plot_states_and_fock_number(1)



.. image:: 08C_Bosonic_grape_with_smooth_pulses_files/08C_Bosonic_grape_with_smooth_pulses_52_0.png


Finally, lets plot the variation of infidelity with evaluation number.

.. code:: ipython3

    from plotting import plot_infidelity_vs_evaluation_from_logs
    
    plot_infidelity_vs_evaluation_from_logs(log_path=temp_dir.name + "/opt.log", label="dCRAB optimization");



.. image:: 08C_Bosonic_grape_with_smooth_pulses_files/08C_Bosonic_grape_with_smooth_pulses_54_0.png


The sharp rise infidelity represents the beginning of a super-iteration

References
----------

| [Sørensen2018]Sørensen, Jens Jakob WH, et al. “Quantum optimal control
  in a chopped basis: Applications in control of Bose-Einstein
  condensates.” Physical Review A 98.2 (2018): 022119.
| [Müller2022] Matthias M Müller et al “One decade of quantum optimal
  control in the chopped random basis”, Rep. Prog. Phys. 85 076001
  (2022)
| [Heeres2017] R. Heeres et al., “Implementing a universal gate set on a
  logical qubit encoded in an oscillator”, Nature Communications 8, 94
  (2017)
