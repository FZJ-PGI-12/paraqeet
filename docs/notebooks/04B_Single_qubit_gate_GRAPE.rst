Single spin Part 3: Single qubit gate optimisation using GRAPE
==============================================================

1. Generate a PWC pulse shape
-----------------------------

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    
    from paraqeet.quantity import Quantity
    from paraqeet.signal.envelopes import GaussEnvelope
    from paraqeet.signal.pwc_generator import PWCGenerator

First, let’s generate a piecewise constant (PWC) pulse envelope for the
Gaussian pulse

.. code:: ipython3

    t_final = 20e-9
    tlist = np.linspace(0, t_final, 101)
    tone = GaussEnvelope(amplitude=Quantity(2 * np.pi / t_final / 3, -5 * np.pi / t_final, 5 * np.pi / t_final))
    tone.t_final.set_value(t_final)
    gen = PWCGenerator(envelopes=[tone], tlist=tlist)
    gen.multiply_flat_top = True

.. code:: ipython3

    params = gen.get_parameters()

.. code:: ipython3

    ts = np.linspace(0, t_final, 1001)
    
    plt.plot(ts / 1e-9, tone.compute_output(ts) / 1e6, label="Smooth curve")
    plt.plot(ts / 1e-9, np.real(gen.generate_signal(ts)) / 1e6, ls="--", label="in-phase")
    plt.plot(
        ts / 1e-9,
        np.imag(gen.generate_signal(ts)) / 1e6,
        ls="--",
        label="out-of-phase",
    )
    
    plt.xlabel("Time [in ns]")
    plt.ylabel("Amplitude [in MHz]")
    plt.legend()




.. parsed-literal::

    <matplotlib.legend.Legend at 0x7f44ec27c770>




.. image:: 04B_Single_qubit_gate_GRAPE_files/04B_Single_qubit_gate_GRAPE_6_1.png


2. Define Hamiltonian in the rotating frame of drive
----------------------------------------------------

Next, we setup the qubit system we want to control. We define the
Hamiltonian in the rotating frame of drive such that the pulse
oscillates slowly to apply GRAPE gradients.

The Hamiltonain in the rotating frame of the drive is given by -

.. math::  H(t) = \big(\omega_q - \omega_d\big) b^\dagger b -\frac{\alpha}{2} (b^\dagger)^2 b^2 + (\epsilon(t) b + \epsilon(t)^* b)

.. code:: ipython3

    from paraqeet.quantity import Quantity
    from paraqeet.model.closed_system import ClosedSystem
    from paraqeet.model.rotating_frame_drive import RotatingFrameDrive
    from paraqeet.model.transmon import Transmon
    
    
    freq = 7.86e9 * 2 * np.pi
    dims = 3
    anharm = -50e6 * 2 * np.pi
    offset = 5e6 * 2 * np.pi
    
    drive_freq = freq + offset
    qubit_freq = freq - drive_freq
    
    Drive = RotatingFrameDrive(gen)
    transmon = Transmon(
        frequency=Quantity(
            qubit_freq,
            1.2 * qubit_freq,
            0.8 * qubit_freq,
            unit="Hz",
            name="Frequency",
        ),
        anharmonicity=Quantity(anharm, 1.2 * anharm, 0.8 * anharm, unit="Hz", name="Anharmonicity"),
        drives=[Drive],
        dimension=dims,
    )
    
    model = ClosedSystem(transmon)

.. code:: ipython3

    from paraqeet.measurement.state_transfer_fidelity import StateTransferFidelityGRAPE
    from paraqeet.propagation.scipy_expm_grape import ScipyExpmGRAPE
    
    
    prop = ScipyExpmGRAPE(model, res=1e9)
    
    init = np.array([[1.0], [0.0], [0]])  # |0>
    target = np.array([[0.0], [1.0], [0]])  # |1>
    
    prop.set_initial_state(init)
    prop.target_state = target
    
    prop.use_schirmer_derivative = True
    
    zeroone = StateTransferFidelityGRAPE(
        propagation=prop,
        initial_state=init,
        target_state=target,
        times=tlist,
    )

.. code:: ipython3

    def plot_states():
        """Plot the states."""
        ts = np.linspace(0, t_final, 1001)
        states = prop.propagate(ts)
        sig = gen.generate_signal(ts)
    
        _, ax = plt.subplots(2, figsize=(4, 4), sharex=True)
        ax[0].plot(ts / 1e-9, np.real(sig), label="I")
        ax[0].plot(ts / 1e-9, np.imag(sig), label="Q")
        ax[0].legend(loc=1)
        ax[0].set_ylabel("Field [MHz]")
        ax[1].plot(ts / 1e-9, np.abs(states)[:, :, 0] ** 2)
        ax[1].set_ylabel("Population")
        ax[-1].set_xlabel("Time [ns]")
        plt.show()
    
    
    plot_states()



.. image:: 04B_Single_qubit_gate_GRAPE_files/04B_Single_qubit_gate_GRAPE_11_0.png


As expected, we get a partial transfer and a low fidelity.

.. code:: ipython3

    zeroone.measure()




.. parsed-literal::

    0.3097264170189858



3. Opimisation
--------------

We define an optimiser and link our fidelity measure as a goal function
and the parameters of the cosine tone and optimise just amplitude and
frequency, as in the state transfer example.

.. code:: ipython3

    from paraqeet.optimisation_map import OptimisationMap
    from paraqeet.optimisers.scipy_optimiser_gradient import ScipyOptimiserGradient
    
    optmap = OptimisationMap()
    optmap.add(gen, params)
    opt = ScipyOptimiserGradient(zeroone, optimisables=optmap)

.. code:: ipython3

    opt.optimise()




.. parsed-literal::

    {'status': 1, 'value': 2.2687367540186187e-09, 'iterations': 14, 'message': 'CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL'}



.. code:: ipython3

    ts = np.linspace(0, t_final, 1001)
    
    plt.plot(ts / 1e-9, np.real(gen.generate_signal(ts)) / 1e6, ls="--", label="in-phase")
    plt.plot(
        ts / 1e-9,
        np.imag(gen.generate_signal(ts)) / 1e6,
        ls="--",
        label="out-of-phase",
    )
    
    plt.xlabel("Time [in ns]")
    plt.ylabel("Amplitude [in MHz]")
    plt.legend()
    plt.show()



.. image:: 04B_Single_qubit_gate_GRAPE_files/04B_Single_qubit_gate_GRAPE_17_0.png


.. code:: ipython3

    plot_states()



.. image:: 04B_Single_qubit_gate_GRAPE_files/04B_Single_qubit_gate_GRAPE_18_0.png

