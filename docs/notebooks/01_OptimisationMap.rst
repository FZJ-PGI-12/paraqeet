Handling parameters in ParaQeet
===============================

The optimisation map is a utility class that collects all parameters
that shall be considered during optimisation and associates them with
the corresponding Optimisable interface. With this class, Quantities can
be traced back to the Optimisable to which they belong. Before
optimisation, an instance of this class needs to be filled and passed to
the optimiser.

.. code:: ipython3

    from paraqeet.optimisation_map import OptimisationMap
    
    from paraqeet.signal.iq_mixer import IQMixer
    from paraqeet.signal.envelopes import ConstantEnvelope, FlatTopGaussianEnvelope

Example devices and their parameters
------------------------------------

We define some signal generator and look at its parameters:

.. code:: ipython3

    tone = ConstantEnvelope()
    gen = IQMixer(envelopes=[tone])

.. code:: ipython3

    params = gen.get_parameters()
    params




.. parsed-literal::

    [Amplitude: 24.7 MHz x 2pi,
     t_final: 32 ns,
     lo_freq: 4.8 GHz x 2pi,
     Phase: 0 rad]



The Optimisation Map
--------------------

To handle the parameters of both tones, we make an OptimisationMap and
add the parameters of the first tone explcitely.

.. code:: ipython3

    optmap = OptimisationMap()
    optmap.add(gen, params)

We can get a list output of all parameters with

.. code:: ipython3

    optmap.get_all_parameters()




.. parsed-literal::

    [Amplitude: 24.7 MHz x 2pi,
     t_final: 32 ns,
     lo_freq: 4.8 GHz x 2pi,
     Phase: 0 rad]



Or a human readalbe output to check that we didn’t make a mistake in
configuration.

.. code:: ipython3

    optmap




.. parsed-literal::

    ==== <class 'paraqeet.signal.iq_mixer.IQMixer'> ====
    [Amplitude: 24.7 MHz x 2pi, t_final: 32 ns, lo_freq: 4.8 GHz x 2pi, Phase: 0 rad]




If we just want to optimise just the frequency, we set

.. code:: ipython3

    optmap.add(gen, [params[2]])

.. code:: ipython3

    optmap




.. parsed-literal::

    ==== <class 'paraqeet.signal.iq_mixer.IQMixer'> ====
    [lo_freq: 4.8 GHz x 2pi]




Adding more devices
-------------------

As a second drive, we create a signal shaped by an error function
envelope:

.. code:: ipython3

    tone2 = FlatTopGaussianEnvelope()
    gen2 = IQMixer(envelopes=[tone2])

If we don’t specify an explicit list of parameters, all of them get
added.

.. code:: ipython3

    optmap.add(gen2)

.. code:: ipython3

    optmap.get_all_parameters()




.. parsed-literal::

    [lo_freq: 4.8 GHz x 2pi,
     Amplitude: 24.7 MHz x 2pi,
     t_final: 32 ns,
     lo_freq: 4.8 GHz x 2pi,
     Phase: 0 rad]



.. code:: ipython3

    optmap




.. parsed-literal::

    ==== <class 'paraqeet.signal.iq_mixer.IQMixer'> ====
    [lo_freq: 4.8 GHz x 2pi]
    
    ==== <class 'paraqeet.signal.iq_mixer.IQMixer'> ====
    [Amplitude: 24.7 MHz x 2pi, t_final: 32 ns, lo_freq: 4.8 GHz x 2pi, Phase: 0 rad]




Selecting parameters
--------------------

There’s a convenient filter method to select parameters based on
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

    optmap




.. parsed-literal::

    ==== <class 'paraqeet.signal.iq_mixer.IQMixer'> ====
    [Amplitude: 24.7 MHz x 2pi]




Adding back all parameters:

.. code:: ipython3

    optmap.add(gen)
    optmap.add(gen2)
    optmap




.. parsed-literal::

    ==== <class 'paraqeet.signal.iq_mixer.IQMixer'> ====
    [Amplitude: 24.7 MHz x 2pi, t_final: 32 ns, lo_freq: 4.8 GHz x 2pi, Phase: 0 rad]
    
    ==== <class 'paraqeet.signal.iq_mixer.IQMixer'> ====
    [Amplitude: 24.7 MHz x 2pi, t_final: 32 ns, lo_freq: 4.8 GHz x 2pi, Phase: 0 rad]




Now, we select every parameter with unit “Hz”:

.. code:: ipython3

    def hz_filter(par):
        """Get every parameter with the unit 'Hz'."""
        return par.get_unit() == "Hz"
    
    
    optmap.filter_parameters(hz_filter)
    optmap




.. parsed-literal::

    ==== <class 'paraqeet.signal.iq_mixer.IQMixer'> ====
    [Amplitude: 24.7 MHz x 2pi, lo_freq: 4.8 GHz x 2pi]
    
    ==== <class 'paraqeet.signal.iq_mixer.IQMixer'> ====
    [Amplitude: 24.7 MHz x 2pi, lo_freq: 4.8 GHz x 2pi]



