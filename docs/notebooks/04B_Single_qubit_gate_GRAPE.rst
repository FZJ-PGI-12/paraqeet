Single spin Part 3: Single qubit gate optimization using GRAPE
==============================================================

1. Generate a PWC pulse shape
-----------------------------

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    
    from paraqeet.model.drive import Drive
    from paraqeet.model.schroedinger_equation import SchroedingerEquation
    from paraqeet.model.transmon import Transmon
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

    from plotting import plot_signal
    
    ts = np.linspace(0, t_final, 501)
    fig, ax = plt.subplots(1, figsize=(5, 3))
    plot_signal(tone, ts, ax, linestyle="-", label="Smooth")
    plot_signal(gen, ts, ax, linestyle="--", label="PWC")
    ax.legend(loc=1, frameon=True)
    plt.show()



.. image:: 04B_Single_qubit_gate_GRAPE_files/04B_Single_qubit_gate_GRAPE_6_0.png


2. Define Hamiltonian in the rotating frame of drive
----------------------------------------------------

Next, we setup the qubit system we want to control. We define the
Hamiltonian in the rotating frame of drive such that the pulse
oscillates slowly to apply GRAPE gradients.

The Hamiltonain in the rotating frame of the drive is given by -

.. math::  H(t) = \big(\omega_q - \omega_d\big) b^\dagger b -\frac{\alpha}{2} (b^\dagger)^2 b^2 + (\epsilon(t) b + \epsilon(t)^* b)

.. code:: ipython3

    freq = 7.86e9 * 2 * np.pi
    num_levels = 3
    anharm = -50e6 * 2 * np.pi
    offset = 5e6 * 2 * np.pi
    
    drive_freq = freq + offset
    qubit_freq = freq - drive_freq
    
    transmon = Transmon(
        frequency=Quantity(
            qubit_freq,
            1.2 * qubit_freq,
            0.8 * qubit_freq,
            unit="Hz",
            name="Frequency",
        ),
        anharmonicity=Quantity(anharm, 1.2 * anharm, 0.8 * anharm, unit="Hz", name="Anharmonicity"),
        drives=[],
        num_levels=num_levels,
    )
    
    drive = Drive(transmon.annihilation_op, gen, add_hermitian=True)
    transmon.drives = [drive]
    
    
    model = SchroedingerEquation(
        hamiltonian_func=transmon.get_value, hamiltonian_and_gradient_func=transmon.get_value_and_gradient
    )

.. code:: ipython3

    from paraqeet.measurement.state_transfer_fidelity import StateTransferFidelityGRAPE
    from paraqeet.measurement.utils import overlap_state_vector
    from paraqeet.propagation.scipy_expm_grape import ScipyExpmGRAPE
    from paraqeet.propagation.utils import grape_operator_sandwich_function_closed
    
    init = np.array([[1.0], [0.0], [0.0]])  # |0>
    target = np.array([[0.0], [1.0], [0.0]])  # |1>
    
    times = np.array([0.0, t_final])
    
    prop = ScipyExpmGRAPE(
        eom_func=model.get_value,
        eom_and_grad_func=model.get_value_and_gradient,
        resolution=1e9,
        initial_state=init,
        target_state=target,
        operator_sandwich_function=grape_operator_sandwich_function_closed,
    )
    prop.schirmer_derivative = True
    
    zeroone = StateTransferFidelityGRAPE(
        propagation_func=prop.propagate,
        propagation_and_gradient_func=prop.get_value_and_gradient,
        target_state=target,
        overlap=overlap_state_vector,
    )

.. code:: ipython3

    from plotting import plot_signal_and_dynamics
    
    ts = np.linspace(0.0, t_final, 101)
    plot_signal_and_dynamics(gen, prop, ts, state_labels=[r"$|0\rangle$", r"$|1\rangle$"])




.. parsed-literal::

    array([<Axes: ylabel='Amplitude \n[MHz / $2\\pi$]'>,
           <Axes: xlabel='Time [ns]', ylabel='Population'>], dtype=object)




.. image:: 04B_Single_qubit_gate_GRAPE_files/04B_Single_qubit_gate_GRAPE_11_1.png


As expected, we get a partial transfer and a low fidelity.

.. code:: ipython3

    zeroone.measure(times)




.. parsed-literal::

    Array(0.74491766, dtype=float64)



3. Optimization
---------------

We define an optimizer and link our fidelity measure as a goal function
and the parameters of the cosine tone and optimize just amplitude and
frequency, as in the state transfer example.

.. code:: ipython3

    from paraqeet.optimization_map import OptimizationMap
    from paraqeet.optimizers.scipy_optimizer_gradient import ScipyOptimizerGradient
    
    optmap = OptimizationMap()
    optmap.add(gen, params)
    opt = ScipyOptimizerGradient(measure_and_gradient_func=zeroone.get_value_and_gradient, optimization_map=optmap)

.. code:: ipython3

    opt.optimize(gen.tlist)


.. parsed-literal::

    Iteration    1 | Infid = 6.132357e-01
    Iteration    2 | Infid = 2.186693e-01
    Iteration    3 | Infid = 4.769458e-02
    Iteration    4 | Infid = 4.665449e-03
    Iteration    5 | Infid = 2.567502e-03
    Iteration    6 | Infid = 1.584818e-03
    Iteration    7 | Infid = 5.730724e-04


.. parsed-literal::

    Iteration    8 | Infid = 1.134631e-05
    Iteration    9 | Infid = 3.978981e-07
    Iteration   10 | Infid = 1.254676e-08
    Iteration   11 | Infid = 2.763889e-10




.. parsed-literal::

    {'status': 1, 'value': 2.763889117574081e-10, 'iterations': 14, 'message': 'CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL'}



.. code:: ipython3

    plot_signal_and_dynamics(gen, prop, ts, state_labels=[r"$|0\rangle$", r"$|1\rangle$"])




.. parsed-literal::

    array([<Axes: ylabel='Amplitude \n[MHz / $2\\pi$]'>,
           <Axes: xlabel='Time [ns]', ylabel='Population'>], dtype=object)




.. image:: 04B_Single_qubit_gate_GRAPE_files/04B_Single_qubit_gate_GRAPE_17_1.png

