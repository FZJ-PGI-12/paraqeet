Handling parameters in ParaQeet
===============================

The ``OptimizationMap`` is a utility class that collects all parameters
that shall be considered during optimization and associates them with
the corresponding ``Optimizable`` interface. With this class,
``Quantity`` objects can be traced back to the ``Optimizable`` to which
they belong. Before optimization, an instance of this class needs to be
filled and passed to the optimizer.

.. code:: ipython3

    from paraqeet.optimizable import Optimizable
    from paraqeet.optimization_map import OptimizationMap
    from paraqeet.quantity import Quantity
    from paraqeet.signal.envelopes import ConstantEnvelope, FlatTopGaussianEnvelope
    from paraqeet.signal.iq_mixer import IQMixer

Example devices and their parameters
------------------------------------

We define a tone and a signal generator and look at their parameters:

.. code:: ipython3

    tone = ConstantEnvelope()
    
    gen = IQMixer(envelopes=[tone])

We note that both ``tone`` and ``gen`` are instances of ``Optimizable``.

.. code:: ipython3

    print(isinstance(tone, Optimizable))
    print(isinstance(gen, Optimizable))


.. parsed-literal::

    True
    True


As such they must have a ``get_parameters`` method implemented. We can
obtain the (default) parameters of the tone as

.. code:: ipython3

    params_tone = tone.get_parameters()
    print(params_tone)


.. parsed-literal::

    [Amplitude: 24.7 MHz x 2pi, t_final: 32 ns]


and those of the generator as

.. code:: ipython3

    params_gen = gen.get_parameters()
    print(params_gen)


.. parsed-literal::

    [Amplitude: 24.7 MHz x 2pi, t_final: 32 ns, lo_freq: 4.8 GHz x 2pi, Phase: 0 rad]


As we see, in this case, ``gen`` has the parameters of the ``tone`` as
its parameters. Moreover, it has two additional parameters, namely, the
frequency of the local oscillator (‘lo_freq’) and the phase of the pulse
(‘Phase’). We also note that the ``get_parameters`` returns a list of
ParaQeet’s ``Quantity`` objects:

.. code:: ipython3

    print(isinstance(params_gen, list))
    print([isinstance(param, Quantity) for param in params_gen])


.. parsed-literal::

    True
    [True, True, True, True]


The Optimization Map
--------------------

To handle the parameters of both tones, we make an
``OptimizationMap``\ and add the parameters of the ``gen`` explicitly.

.. code:: ipython3

    optmap = OptimizationMap()
    optmap.add(gen, params_gen)

Note that we also need to pass ``gen`` itself when we add the
corresponding parameters to the optmap. We can get a list output of all
parameters with

.. code:: ipython3

    optmap.get_all_parameters()




.. parsed-literal::

    [Amplitude: 24.7 MHz x 2pi,
     t_final: 32 ns,
     lo_freq: 4.8 GHz x 2pi,
     Phase: 0 rad]



If we just want to optimize the frequency, we set

.. code:: ipython3

    optmap.add(gen, [params_gen[2]])
    optmap.get_all_parameters()




.. parsed-literal::

    [lo_freq: 4.8 GHz x 2pi]



which overwrites the previous parameters associated with ``gen``.

Adding more devices
-------------------

As a second drive, we create a signal shaped by a flat-top Gaussian
envelope:

.. code:: ipython3

    tone_2 = FlatTopGaussianEnvelope()
    gen_2 = IQMixer(envelopes=[tone_2])

If we don’t specify an explicit list of parameters, all of them get
added.

.. code:: ipython3

    optmap.add(gen_2)
    optmap.get_all_parameters()




.. parsed-literal::

    [lo_freq: 4.8 GHz x 2pi,
     Amplitude: 24.7 MHz x 2pi,
     t_up: 6.4 ns,
     t_down: 25.6 ns,
     ramp_time: 3.2 ns,
     lo_freq: 4.8 GHz x 2pi,
     Phase: 0 rad]



If we just print ``optmap`` we can also see that the parameters are
associated with different objects

.. code:: ipython3

    print(optmap)


.. parsed-literal::

    ==== <class 'paraqeet.signal.iq_mixer.IQMixer'> ====
    [lo_freq: 4.8 GHz x 2pi]
    
    ==== <class 'paraqeet.signal.iq_mixer.IQMixer'> ====
    [Amplitude: 24.7 MHz x 2pi, t_up: 6.4 ns, t_down: 25.6 ns, ramp_time: 3.2 ns, lo_freq: 4.8 GHz x 2pi, Phase: 0 rad]
    
    


Selecting parameters
--------------------

There is a convenient filter method to select parameters based on
properties. The following example selects all amplitudes:

.. code:: ipython3

    def all_amplitudes(par):
        """Get the amplitudes."""
        return par.get_name() == "Amplitude"
    
    
    optmap.filter_parameters(all_amplitudes)
    optmap.get_all_parameters()




.. parsed-literal::

    [Amplitude: 24.7 MHz x 2pi]



.. code:: ipython3

    print(optmap)


.. parsed-literal::

    ==== <class 'paraqeet.signal.iq_mixer.IQMixer'> ====
    [Amplitude: 24.7 MHz x 2pi]
    
    


Adding back all parameters:

.. code:: ipython3

    optmap.add(gen)
    optmap.add(gen_2)
    print(optmap)


.. parsed-literal::

    ==== <class 'paraqeet.signal.iq_mixer.IQMixer'> ====
    [Amplitude: 24.7 MHz x 2pi, t_up: 6.4 ns, t_down: 25.6 ns, ramp_time: 3.2 ns, lo_freq: 4.8 GHz x 2pi, Phase: 0 rad]
    
    ==== <class 'paraqeet.signal.iq_mixer.IQMixer'> ====
    [Amplitude: 24.7 MHz x 2pi, t_final: 32 ns, lo_freq: 4.8 GHz x 2pi, Phase: 0 rad]
    
    


Now, we select every parameter with unit “Hz”:

.. code:: ipython3

    def hz_filter(par):
        """Get every parameter with the unit 'Hz'."""
        return par.get_unit() == "Hz"
    
    
    optmap.filter_parameters(hz_filter)
    print(optmap)


.. parsed-literal::

    ==== <class 'paraqeet.signal.iq_mixer.IQMixer'> ====
    [Amplitude: 24.7 MHz x 2pi, lo_freq: 4.8 GHz x 2pi]
    
    ==== <class 'paraqeet.signal.iq_mixer.IQMixer'> ====
    [Amplitude: 24.7 MHz x 2pi, lo_freq: 4.8 GHz x 2pi]
    
    


In the following notebooks we will put to practice what we learned in
this notebook and use the ``OptimizationMap`` to select the parameters
to optimize in optimal control tasks.
