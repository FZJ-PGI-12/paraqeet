Single qubit gate optimization using GOAT over GRAPE
====================================================

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    
    from paraqeet.signal.pwc_generator import PWCGenerator
    from paraqeet.signal.waveform import DRAGMixer, FlatTopGaussianFilter

Setup
-----

Define Hamiltonian in the rotating frame of drive Next, we setup the
qubit system we want to control. We define the Hamiltonian in the
rotating frame of drive such that the pulse oscillates slowly to apply
GRAPE gradients.

The Hamiltonain in the rotating frame of the drive is given by -

.. math::  H(t) = \big(\omega_q - \omega_d\big) b^\dagger b -\frac{\alpha}{2} (b^\dagger)^2 b^2 + (\epsilon(t) b + \epsilon(t)^* b)

.. code:: ipython3

    from paraqeet.model.closed_system import ClosedSystem
    from paraqeet.model.rotating_frame_drive import RotatingFrameDrive
    from paraqeet.model.transmon import Transmon
    from paraqeet.quantity import Quantity
    
    freq = 4e9 * 2 * np.pi
    dims = 3
    anharm = -200e6 * 2 * np.pi
    offset = 2e6 * 2 * np.pi
    
    drive_freq = freq + offset
    qubit_freq = freq - drive_freq

.. code:: ipython3

    from paraqeet.signal.envelopes import FlatTopGaussianEnvelope
    
    t_final = 25e-9
    tlist = np.linspace(0, t_final, 151)
    
    tone = FlatTopGaussianEnvelope(
        amplitude=Quantity(np.pi / t_final / 3, -np.pi / t_final, np.pi / t_final, name="Amplitude"),
        t_up=Quantity(1e-9, 0.0, t_final, name="t_up"),
        t_down=Quantity(t_final - 1e-9, 0.0, t_final, name="t_down"),
        ramp_time=Quantity(2e-9, 0.5e-9, t_final, name="ramp_time"),
        t_final=Quantity(t_final, 0.9 * t_final, 1, 1 * t_final, name="t_final"),
    )
    
    drag_tone = DRAGMixer(
        tone,
        deltas=[Quantity(0.5 * anharm, min_value=3 * anharm, max_value=anharm / 3, unit="Hz", two_pi=True, name="Delta")],
    )
    
    filtered_tone = FlatTopGaussianFilter(drag_tone, t_final=Quantity(t_final, 0.0, 1.2 * t_final, name="t_final"))
    
    gen = PWCGenerator(envelopes=[filtered_tone], tlist=tlist)

.. code:: ipython3

    params = drag_tone.get_parameters()
    params




.. parsed-literal::

    [Amplitude: 4.19e+07,
     t_up: 1e-09,
     t_down: 2.4e-08,
     ramp_time: 2e-09,
     Delta: -100 MHz x 2pi]



.. code:: ipython3

    pwc_signal = gen.get_parameters()
    pwc_signal[0].set_limits(-4e9, 4e9)
    pwc_signal[1].set_limits(-4e9, 4e9)

.. code:: ipython3

    from plotting import plot_signal
    
    ts = np.linspace(0, t_final, 501)
    fig, ax = plt.subplots(1, figsize=(5, 3))
    plot_signal(filtered_tone, ts, ax, linestyle="-", label="Smooth")
    plot_signal(gen, ts, ax, linestyle="--", label="PWC")
    ax.legend(loc=1, frameon=True)
    plt.show()



.. image:: 04C_Single_qubit_gate_GOAToverGRAPE_files/04C_Single_qubit_gate_GOAToverGRAPE_7_0.png


.. code:: ipython3

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
    
    prop = ScipyExpmGRAPE(model, resolution=1e9)
    
    init = np.array([[1.0], [0.0], [0]])  # |0>
    target = np.array([[0.0], [1.0], [0]])  # |1>
    
    prop.set_initial_state(init)
    prop.set_target_state(target)
    times = np.array([0.0, t_final])
    
    prop.use_schirmer_derivative = True
    
    zeroone = StateTransferFidelityGRAPE(
        propagation=prop,
        initial_state=init,
        target_state=target,
    )

.. code:: ipython3

    from plotting import plot_signal_and_dynamics
    
    ts = np.linspace(0.0, t_final, 101)
    plot_signal_and_dynamics(gen, prop, ts, state_labels=[r"$|0\rangle$", r"$|1\rangle$"]);



.. image:: 04C_Single_qubit_gate_GOAToverGRAPE_files/04C_Single_qubit_gate_GOAToverGRAPE_10_0.png


As expected, we get a partial transfer and a low fidelity.

.. code:: ipython3

    zeroone.measure(times)




.. parsed-literal::

    0.7434255669360508



We define an optimizer and link our fidelity measure as a goal function
and the parameters of the cosine tone and optimize just amplitude and
frequency, as in the state transfer example.

.. code:: ipython3

    drag_tone.get_parameters()




.. parsed-literal::

    [Amplitude: 4.19e+07,
     t_up: 1e-09,
     t_down: 2.4e-08,
     ramp_time: 2e-09,
     Delta: -100 MHz x 2pi]



Optimization
------------

.. code:: ipython3

    from paraqeet.measurement.goat_over_grape import GOATOverGRAPE
    from paraqeet.optimization_map import OptimizationMap
    from paraqeet.optimizers.scipy_optimizer import ScipyOptimizer
    from paraqeet.optimizers.scipy_optimizer_gradient import ScipyOptimizerGradient
    
    optmap = OptimizationMap()
    optmap.add(filtered_tone)
    optmap.register_params_with_optimizables()
    
    goat = GOATOverGRAPE(zeroone, propagation=prop, generators=[gen])
    
    opt = ScipyOptimizer(goat, optimization_map=optmap)
    optgrad = ScipyOptimizerGradient(goat, optimization_map=optmap)

.. code:: ipython3

    optmap




.. parsed-literal::

    ==== <class 'paraqeet.signal.waveform.FlatTopGaussianFilter'> ====
    [Amplitude: 4.19e+07, t_up: 1e-09, t_down: 2.4e-08, ramp_time: 2e-09, Delta: -100 MHz x 2pi]




.. code:: ipython3

    optgrad.optimize(np.array([0.0, t_final]))




.. parsed-literal::

    {'status': 1, 'value': 0.0026361558796704765, 'iterations': 46, 'message': 'CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH'}



.. code:: ipython3

    plot_signal_and_dynamics(gen, prop, ts, state_labels=[r"$|0\rangle$", r"$|1\rangle$"]);



.. image:: 04C_Single_qubit_gate_GOAToverGRAPE_files/04C_Single_qubit_gate_GOAToverGRAPE_19_0.png

