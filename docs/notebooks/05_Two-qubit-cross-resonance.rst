Gradient-based optimisation of a cross-resonance gate between two transmons
===========================================================================

.. code:: ipython3

    import itertools
    import matplotlib.pyplot as plt
    import numpy as np
    
    from paraqeet.signal.envelopes import FlatTopGaussianEnvelope
    from paraqeet.signal.iq_mixer import IQMixer
    from paraqeet.propagation.scipy_expm_goat import ScipyExpmGOAT
    from paraqeet.propagation.propagation import Propagation
    from paraqeet.model.drive_operator import DriveOperator
    from paraqeet.model.coupling import Coupling
    from paraqeet.model.closed_system import ClosedSystem
    from paraqeet.model.composite_hamiltonian import CompositeHamiltonian
    from paraqeet.model.transmon import Transmon
    from paraqeet.optimisers.scipy_optimiser_gradient import ScipyOptimiserGradient
    from paraqeet.measurement.unitary_fidelity import UnitaryFidelity
    from paraqeet.optimisation_map import OptimisationMap
    from paraqeet.quantity import Quantity
    
    np.set_printoptions(linewidth=400)

System Setup
------------

The sytem consists of two coupled transmons with three levels each. We
fix the transmon frequency and anharmonicity to values that don’t have
any unwanted frequency collisions. The coupling strength is fixed as
well. These parameters have to be specified as Quantites with a range,
but we will not pass them to the optimised in order to keep them fixed.
Additionally, the first transmon is driven at the frequency of the
second one to apply a cross-resonance (CR) gate. The second transmon is
driven to fix the phases of the gate.

Here the tone values are set such that the optimisation process is fast.
Generally with a lot of parameters ``ScipyExpmGOAT`` (in its current
form), can take considerably long time.

.. code:: ipython3

    t_final = 150e-9
    tone1 = FlatTopGaussianEnvelope(
        amplitude=Quantity(
            190e6 * 2 * np.pi, min_value=1e5 * 2 * np.pi, max_value=250e6 * 2 * np.pi, name="Amp", unit="Hz", two_pi=True
        ),
        t_final=Quantity(
            t_final,
            min_value=10e-9,
            max_value=200e-9,
            unit="s",
            name="Gate time",
        ),
    )
    
    tone2 = FlatTopGaussianEnvelope(
        amplitude=Quantity(
            9.18e6 * 2 * np.pi,
            min_value=1e5 * 2 * np.pi,
            max_value=250e6 * 2 * np.pi,
            name="Amp",
            unit="Hz",
            two_pi=True,
        ),
        t_final=Quantity(
            t_final,
            min_value=10e-9,
            max_value=200e-9,
            unit="s",
            name="Gate time",
        ),
    )
    
    
    tone2.t_final = Quantity(
        t_final,
        min_value=10e-9,
        max_value=200e-9,
        unit="s",
        name="Gate time",
    )
    
    print(tone1.get_parameters(), tone2.get_parameters())
    generator1 = IQMixer(
        envelopes=[tone1],
        frequency=Quantity(
            6.0002e9 * 2 * np.pi,
            min_value=5.5e9 * 2 * np.pi,
            max_value=6.20e9 * 2 * np.pi,
            name="Freq",
            unit="Hz",
            two_pi=True,
        ),
    )
    drive1 = DriveOperator(generator1, is_longitudinal=False)
    
    generator2 = IQMixer(
        envelopes=[tone2],
        frequency=Quantity(
            6.0002e9 * 2 * np.pi,
            min_value=5.5e9 * 2 * np.pi,
            max_value=6.20e9 * 2 * np.pi,
            name="Freq",
            unit="Hz",
            two_pi=True,
        ),
        phase=Quantity(
            value=-0.39720756,
            min_value=-np.pi,
            max_value=np.pi,
            unit="rad",
            name="Phase",
        ),
    )
    
    drive2 = DriveOperator(generator2, is_longitudinal=False)
    
    transmon1 = Transmon(
        dimension=3,
        frequency=Quantity(
            5.5e9 * 2 * np.pi,
            5.2e9 * 2 * np.pi,
            5.9e9 * 2 * np.pi,
            "Hz",
            "Transmon 1 frequency",
            two_pi=True,
        ),
        anharmonicity=Quantity(
            -240e6 * 2 * np.pi,
            -250e6 * 2 * np.pi,
            -190e6 * 2 * np.pi,
            "Hz",
            "Transmon 1 anharmonicity",
            two_pi=True,
        ),
        drives=[drive1],
    )
    transmon2 = Transmon(
        dimension=3,
        frequency=Quantity(
            6.0e9 * 2 * np.pi,
            5.9e9 * 2 * np.pi,
            6.1e9 * 2 * np.pi,
            "Hz",
            "Transmon 2 frequency",
            two_pi=True,
        ),
        anharmonicity=Quantity(
            -200e6 * 2 * np.pi,
            -210e6 * 2 * np.pi,
            -190e6 * 2 * np.pi,
            "Hz",
            "Transmon 2 anharmonicity",
            two_pi=True,
        ),
        drives=[drive2],
    )
    coupling = Coupling(
        [transmon1, transmon2],
        is_longitudinal=False,
        coefficient=Quantity(
            25e6 * 2 * np.pi,
            10e6 * 2 * np.pi,
            60e6 * 2 * np.pi,
            "Hz",
            "Coupling strength",
            two_pi=True,
        ),
    )
    hamiltonian = CompositeHamiltonian([transmon1, transmon2], [coupling])


.. parsed-literal::

    [Amp: 190 MHz x 2pi, Gate time: 150 ns] [Amp: 9.18 MHz x 2pi, Gate time: 150 ns]


.. code:: ipython3

    (transmon2.frequency.get_value() - transmon1.frequency.get_value()) / (2 * np.pi)




.. parsed-literal::

    Array([5.e+08], dtype=float64)



.. code:: ipython3

    from paraqeet.plotting import plot_signal
    
    times = np.linspace(0, t_final, 201)
    fig, ax = plt.subplots(1, figsize=(5, 3))
    plot_signal(tone1, times, ax, linestyle="-", label="Drive 1")
    plot_signal(tone2, times, ax, linestyle="-", label="Drive 2")
    ax.legend(loc=1, frameon=True)
    plt.show()



.. image:: 05_Two-qubit-cross-resonance_files/05_Two-qubit-cross-resonance_6_0.png


We check the eigenvalues to make sure that there is no resonance while
idling.

.. code:: ipython3

    matrix = hamiltonian.get_matrix_one_time(0.0)
    evals = np.linalg.eigvalsh(matrix)
    print("Energies in GHz: ", np.round(evals / 1e6) / 1e3 / (2 * np.pi))
    transitions = evals[1:] - evals[:-1]
    print(
        "Transition energies in GHz: ",
        np.round(transitions / 1e3) / 1e6 / (2 * np.pi),
    )
    matrix


.. parsed-literal::

    Energies in GHz:  [-0.          5.49864413  6.00109628 10.75823753 11.49735309 11.80404466 16.7555141  17.30475781 22.56021318]
    Transition energies in GHz:  [5.49869649 0.50249417 4.75717547 0.73908102 0.3067312  4.95139654 0.54918068 5.25552493]




.. parsed-literal::

    Array([[0.00000000e+00, 1.24401983e+05, 0.00000000e+00, 2.79215219e+06, 1.57079633e+08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
           [1.24401983e+05, 3.76991118e+10, 1.75930971e+05, 1.57079633e+08, 2.79215219e+06, 2.22144147e+08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
           [0.00000000e+00, 1.75930971e+05, 7.41415866e+10, 0.00000000e+00, 2.22144147e+08, 2.79215219e+06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
           [2.79215219e+06, 1.57079633e+08, 0.00000000e+00, 3.45575192e+10, 1.24401983e+05, 0.00000000e+00, 3.94869950e+06, 2.22144147e+08, 0.00000000e+00],
           [1.57079633e+08, 2.79215219e+06, 2.22144147e+08, 1.24401983e+05, 7.22566310e+10, 1.75930971e+05, 2.22144147e+08, 3.94869950e+06, 3.14159265e+08],
           [0.00000000e+00, 2.22144147e+08, 2.79215219e+06, 0.00000000e+00, 1.75930971e+05, 1.08699106e+11, 0.00000000e+00, 3.14159265e+08, 3.94869950e+06],
           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.94869950e+06, 2.22144147e+08, 0.00000000e+00, 6.76070739e+10, 1.24401983e+05, 0.00000000e+00],
           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.22144147e+08, 3.94869950e+06, 3.14159265e+08, 1.24401983e+05, 1.05306186e+11, 1.75930971e+05],
           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.14159265e+08, 3.94869950e+06, 0.00000000e+00, 1.75930971e+05, 1.41748661e+11]], dtype=float64)



Reducing to subsystems
----------------------

We select a propagation method, piecewise constant exponentation, and
configure CR as a target gate. Also we initialize the full basis at time
0 with :math:`\mathbb{I}_4` in order to compute the full propagator. The
``restrict_subsystems`` method has been used to truncate the Hilbert
space to 4 dimensions to compare the propagator to the ideal gate.

.. code:: ipython3

    model = ClosedSystem(hamiltonian)
    prop = ScipyExpmGOAT(model, res=100e9)
    
    pauli_x = np.array([[0.0, 1], [1, 0.0]])
    pauli_y = np.array([[0.0, -1j], [1j, 0.0]])
    pauli_z = np.array([[1, 0], [0.0, -1]])
    
    pauli_ix = np.kron(np.identity(2), pauli_x)
    pauli_iy = np.kron(np.identity(2), pauli_y)
    pauli_iz = np.kron(np.identity(2), pauli_z)
    
    pauli_zx = np.exp(1j * np.pi / 4) * np.kron(pauli_z, pauli_x)
    
    cr_gate = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0], [0, 0, 1.0, 0]])
    
    cr_gate = pauli_zx @ cr_gate
    
    prop.set_initial_state(np.identity(transmon1.dimension() * transmon2.dimension()))
    gate_fid = UnitaryFidelity(
        propagation=prop,
        gate=cr_gate,
        times=np.array([0.0, t_final]),
    )
    gate_fid.restrict_subsystems([transmon1.dimension(), transmon2.dimension()], [2, 2])
    gate_fid.measure()




.. parsed-literal::

    0.8410664317253163



.. code:: ipython3

    def plot_population(propagation: Propagation):
        """Plot the population from the Propagation object."""
        basis1 = [i for i in range(transmon1.dimension())]
        basis2 = [i for i in range(transmon2.dimension())]
        labels = [rf"$|{i},{j}\rangle$" for (i, j) in itertools.product(basis1, basis2)]
    
        signal1 = generator1.generate_signal(times)
        signal2 = generator2.generate_signal(times)
        states = propagation.propagate(times)
    
        _, ax = plt.subplots(3, figsize=(4, 6), sharex=True)
        ax[0].plot(times / 1e-9, signal1)
        ax[0].plot(times / 1e-9, signal2)
        ax[0].set_xlabel("Time [ns]")
        ax[0].set_ylabel("Signal [Hz]")
        ax[0].grid(True, linestyle=(1, (1, 5)), linewidth=1)
    
        ax[1].plot(times / 1e-9, np.abs(states)[:, :, 0] ** 2, label=labels)
        ax[1].set_xlabel("Time [ns]")
        ax[1].set_ylabel("Population")
        ax[1].legend(ncols=2)
        ax[1].grid(True, linestyle=(1, (1, 5)), linewidth=1)
    
        ax[2].plot(
            times / 1e-9,
            np.abs(states)[:, :, transmon2.dimension()] ** 2,
            label=labels,
        )
        ax[2].set_xlabel("Time [ns]")
        ax[2].set_ylabel("Population")
        ax[2].legend(ncols=2)
        ax[2].grid(True, linestyle=(1, (1, 5)), linewidth=1)
    
        plt.tight_layout()
        plt.show()
    
    
    plot_population(prop)



.. image:: 05_Two-qubit-cross-resonance_files/05_Two-qubit-cross-resonance_11_0.png


.. code:: ipython3

    def expecation_value(Op, states):
        """Get the expected value."""
        ex = []
        for state in states:
            state = gate_fid._preprocess_vector(state)
            ex.append(np.real(state.conj() @ Op @ state.T))
        return ex
    
    
    def plot_pauli():
        """Plot the Pauli operators."""
        states = prop.propagate(times)
        sig1 = generator1.generate_signal(times)
        sig2 = generator2.generate_signal(times)
    
        fig, ax = plt.subplots(3, figsize=(4, 6), sharex=True)
        ax[0].plot(times / 1e-9, sig1)
        ax[0].plot(times / 1e-9, sig2)
        ax[0].set_ylabel("Field [MHz]")
        ax[0].grid(True, linestyle=(1, (1, 5)), linewidth=1)
    
        ax[1].plot(times / 1e-9, expecation_value(pauli_ix, states[:, :, 0]))
        ax[1].plot(times / 1e-9, expecation_value(pauli_iy, states[:, :, 0]))
        ax[1].plot(times / 1e-9, expecation_value(pauli_iz, states[:, :, 0]))
        ax[1].set_ylabel(r"$\langle 0, x|\hat\sigma_i|0, x\rangle$")
        ax[1].legend(["X", "Y", "Z"])
        ax[1].grid(True, linestyle=(1, (1, 5)), linewidth=1)
    
        ax[2].plot(times / 1e-9, expecation_value(pauli_ix, states[:, :, transmon2.dimension()]))
        ax[2].plot(times / 1e-9, expecation_value(pauli_iy, states[:, :, transmon2.dimension()]))
        ax[2].plot(times / 1e-9, expecation_value(pauli_iz, states[:, :, transmon2.dimension()]))
        ax[2].set_ylabel(r"$\langle 1, x|\hat\sigma_i|1, x\rangle$")
        ax[-1].set_xlabel("Time [ns]")
        ax[2].legend(["X", "Y", "Z"])
        ax[2].grid(True, linestyle=(1, (1, 5)), linewidth=1)
    
        return fig, ax
    
    
    plot_pauli()




.. parsed-literal::

    (<Figure size 500x750 with 3 Axes>,
     array([<Axes: ylabel='Field [MHz]'>, <Axes: ylabel='$\\langle 0, x|\\hat\\sigma_i|0, x\\rangle$'>, <Axes: xlabel='Time [ns]', ylabel='$\\langle 1, x|\\hat\\sigma_i|1, x\\rangle$'>], dtype=object))




.. image:: 05_Two-qubit-cross-resonance_files/05_Two-qubit-cross-resonance_12_1.png


Optimisation
------------

We define an optimizer and link our fidelity measure as a goal function.
The only optimisable parameter is the frequency of transmon 1.

.. code:: ipython3

    tone1.get_parameters()




.. parsed-literal::

    [Amp: 190 MHz x 2pi, Gate time: 150 ns]



.. code:: ipython3

    optmap = OptimisationMap()
    optmap.add(tone1)
    print(optmap)
    
    opt = ScipyOptimiserGradient(gate_fid, optimisation_map=optmap)
    opt.set_options({"ftol": 0.1})


.. parsed-literal::

    ==== <class 'paraqeet.signal.envelopes.FlatTopGaussianEnvelope'> ====
    [Amp: 190 MHz x 2pi, Gate time: 150 ns]
    
    


.. code:: ipython3

    optmap




.. parsed-literal::

    ==== <class 'paraqeet.signal.envelopes.FlatTopGaussianEnvelope'> ====
    [Amp: 190 MHz x 2pi, Gate time: 150 ns]




.. code:: ipython3

    opt.optimise()




.. parsed-literal::

    {'status': 1, 'value': 0.10112855762826345, 'iterations': 4, 'message': 'CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH'}



.. code:: ipython3

    plot_population(prop)



.. image:: 05_Two-qubit-cross-resonance_files/05_Two-qubit-cross-resonance_18_0.png


.. code:: ipython3

    plot_pauli()




.. parsed-literal::

    (<Figure size 500x750 with 3 Axes>,
     array([<Axes: ylabel='Field [MHz]'>, <Axes: ylabel='$\\langle 0, x|\\hat\\sigma_i|0, x\\rangle$'>, <Axes: xlabel='Time [ns]', ylabel='$\\langle 1, x|\\hat\\sigma_i|1, x\\rangle$'>], dtype=object))




.. image:: 05_Two-qubit-cross-resonance_files/05_Two-qubit-cross-resonance_19_1.png

