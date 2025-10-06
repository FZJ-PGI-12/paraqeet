Single spin: Bayesian optimization of a gate
============================================

This is similar to the OC_Single_qubit_gate example, except that it uses
Bayesian optimization instead of gradient descent.

.. code:: ipython3

    import matplotlib.pyplot as plt
    
    import numpy as np
    from jax import Array
    
    from paraqeet.logger import Logger
    from paraqeet.optimization_map import OptimizationMap
    from paraqeet.quantity import Quantity
    from paraqeet.measurement.unitary_fidelity import UnitaryFidelity
    from paraqeet.model.drive_operator import DriveOperator
    from paraqeet.model.qubit import Qubit
    from paraqeet.optimizers.bayesian_optimizer import BayesianOptimizer
    from paraqeet.propagation.scipy_expm_goat import ScipyExpmGOAT
    
    from paraqeet.model.closed_system import ClosedSystem
    
    from paraqeet.signal.iq_mixer import IQMixer
    from paraqeet.signal.envelopes import ConstantEnvelope

Setup
-----

For signal generation, we define a simple cosine shaped tone generator
:math:`A \cos(\omega t)`

.. code:: ipython3

    t_final = 3e-9
    tone = ConstantEnvelope()
    tone.t_final.set_value(t_final)
    gen = IQMixer(envelopes=[tone])

We can inspect the pre-defined parameters with

.. code:: ipython3

    params = gen.get_parameters()

in this case amplitude :math:`A` and frequency :math:`\omega`.

Next, we setup the qubit system we want to control. We set the qubit
frequency :math:`\omega_q` to be 4.8 GHz and define the Hamiltonian as

.. math:: H(t)=H_\text{drift}+H_c(t)= \frac{\omega_q}{2} \sigma_z + \Omega(t)\sigma_x

, where :math:`\Omega(t)` will be supplied by the generator.

.. code:: ipython3

    freq = 4.8e9 * 2 * np.pi
    
    drive = DriveOperator(gen, is_longitudinal=False)
    controlled_qubit = Qubit(frequency=Quantity(freq, 0.8 * freq, 1.2 * freq), drives=[drive])
    model = ClosedSystem(controlled_qubit)

Textbook values for implementing an :math:`X` rotation on this system at
a time :math:`T` would be :math:`\omega=\omega_q` and :math:`A=\pi/T`.
We use some offset from these values as initial guess to demonstrate the
optimization procedure.

.. code:: ipython3

    params[0].set_value(0.5 * np.pi / t_final)
    params[2].set_value(1.01 * freq)

.. code:: ipython3

    params




.. parsed-literal::

    [Amplitude: 83.3 MHz x 2pi ,
     t_final: 3 ns ,
     lo_freq: 4.85 GHz x 2pi ,
     Phase: 0 rad ]



We select a propagation method, piecewise constant exponentation, and
configure :math:`\sigma_x` as a target gate. Also we initialize the full
basis at time 0 with :math:`\mathcal{I}_2`

.. code:: ipython3

    prop = ScipyExpmGOAT(model, res=500e9)
    
    pauli_x = np.array([[0.0, 1], [1, 0.0]])
    
    prop.set_initial_state(np.identity(2))
    gate_fid = UnitaryFidelity(
        propagation=prop,
        gate=pauli_x,
        times=np.array([0.0, t_final]),
    )

.. code:: ipython3

    def plot_time_evolution():
        """Plot the time evolution of the system."""
        ts = np.linspace(0, t_final, 1001)
        states = prop.propagate(ts)
        sig = gen.generate_signal(ts)
    
        fig, ax = plt.subplots(2, figsize=(4, 4), sharex=True)
        ax[0].plot(ts / 1e-9, sig)
        ax[0].set_ylabel("Field [MHz]")
        ax[1].plot(ts / 1e-9, np.abs(states)[:, :, 0] ** 2)
        ax[1].set_ylabel("Population")
        ax[-1].set_xlabel("Time [ns]")
        return fig, ax
    
    
    plot_time_evolution()




.. parsed-literal::

    (<Figure size 400x400 with 2 Axes>,
     array([<Axes: ylabel='Field [MHz]'>,
            <Axes: xlabel='Time [ns]', ylabel='Population'>], dtype=object))




.. image:: 02C_Qubit-bayesian-optimization_files/02C_Qubit-bayesian-optimization_13_1.png


As expected, we get a partial transfer and a low fidelity.

.. code:: ipython3

    gate_fid.measure()




.. parsed-literal::

    0.008892622545869415



Custom logger implementation
----------------------------

We define an optimizer and link our fidelity measure as a goal function
and the parameters of the cosine tone. We also use a custom logger class
to collect all samples that the optimizer takes

.. code:: ipython3

    samples = []
    
    
    class CustomLogger(Logger):
        """Custom logger class definition."""
    
        def log(self, params: list[Quantity], infid: Array):
            """Log the list of quantities and the fidelity.
    
            Parameters
            ----------
            params: list[Quantity]
                List of parameters of the system.
            infid: Array
                Inverse of fidelity.
    
            """
            samples.append((params[0].get_value(), params[1].get_value()))
    
    
    optmap = OptimizationMap()
    optmap.add(gen, [params[0], params[2], params[3]])
    opt = BayesianOptimizer(gate_fid, optimizables=optmap, initial_samples=10, iterations=100)
    opt.logger = CustomLogger()

.. code:: ipython3

    opt.optimize()


.. parsed-literal::

    |   iter    |  target   |     0     |     1     |     2     |
    -------------------------------------------------------------
    | [39m1        [39m | [39m0.0036571[39m | [39m-0.165955[39m | [39m0.4406489[39m | [39m-0.999771[39m |
    | [39m2        [39m | [39m0.0001105[39m | [39m-0.395334[39m | [39m-0.706488[39m | [39m-0.815322[39m |
    | [39m3        [39m | [39m0.0003969[39m | [39m-0.627479[39m | [39m-0.308878[39m | [39m-0.206465[39m |
    | [35m4        [39m | [35m0.0095249[39m | [35m0.0776334[39m | [35m-0.161610[39m | [35m0.3704390[39m |
    | [39m5        [39m | [39m-.107e-06[39m | [39m-0.591095[39m | [39m0.7562348[39m | [39m-0.945224[39m |
    | [35m6        [39m | [35m0.2036608[39m | [35m0.3409350[39m | [35m-0.165390[39m | [35m0.1173796[39m |
    | [39m7        [39m | [39m0.0007858[39m | [39m-0.719226[39m | [39m-0.603797[39m | [39m0.6014891[39m |
    | [39m8        [39m | [39m0.0438829[39m | [39m0.9365231[39m | [39m-0.373151[39m | [39m0.3846452[39m |


.. parsed-literal::

    | [39m9        [39m | [39m-.744e-06[39m | [39m0.7527783[39m | [39m0.7892133[39m | [39m-0.829911[39m |
    | [39m10       [39m | [39m-.607e-06[39m | [39m-0.921890[39m | [39m-0.660339[39m | [39m0.7562850[39m |


.. parsed-literal::

    | [35m11       [39m | [35m0.2436705[39m | [35m0.3864240[39m | [35m-0.166255[39m | [35m0.0746289[39m |


.. parsed-literal::

    | [35m12       [39m | [35m0.3152984[39m | [35m0.4014301[39m | [35m-0.072420[39m | [35m-0.063436[39m |
    | [39m13       [39m | [39m0.1466834[39m | [39m0.5649364[39m | [39m-0.212259[39m | [39m-0.332136[39m |


.. parsed-literal::

    | [39m14       [39m | [39m0.2810023[39m | [39m0.5635366[39m | [39m0.1276022[39m | [39m0.0006607[39m |
    | [39m15       [39m | [39m0.0152823[39m | [39m0.2674978[39m | [39m0.2561853[39m | [39m-0.169559[39m |
    | [35m16       [39m | [35m0.3836315[39m | [35m0.4408709[39m | [35m-0.098842[39m | [35m-0.084145[39m |


.. parsed-literal::

    | [35m17       [39m | [35m0.4766822[39m | [35m0.6119481[39m | [35m-0.095131[39m | [35m-0.052578[39m |
    | [39m18       [39m | [39m0.2890354[39m | [39m0.4034507[39m | [39m-0.068209[39m | [39m-0.066917[39m |


.. parsed-literal::

    | [35m19       [39m | [35m0.4964185[39m | [35m0.5297423[39m | [35m-0.121980[39m | [35m-0.029561[39m |
    | [39m20       [39m | [39m0.2971542[39m | [39m0.5591593[39m | [39m-0.177524[39m | [39m-0.092768[39m |


.. parsed-literal::

    | [39m21       [39m | [39m0.4609845[39m | [39m0.5715910[39m | [39m-0.053883[39m | [39m0.0224457[39m |
    | [39m22       [39m | [39m0.4405435[39m | [39m0.6169550[39m | [39m-0.144028[39m | [39m0.0384503[39m |


.. parsed-literal::

    | [39m23       [39m | [39m0.0011515[39m | [39m0.9867827[39m | [39m-0.644633[39m | [39m0.8585118[39m |
    | [39m24       [39m | [39m-.931e-05[39m | [39m-0.472166[39m | [39m-0.869049[39m | [39m-0.553717[39m |
    | [39m25       [39m | [39m-.177e-08[39m | [39m-0.877018[39m | [39m-0.657881[39m | [39m0.0207137[39m |
    | [39m26       [39m | [39m0.0012021[39m | [39m-0.795254[39m | [39m-0.508318[39m | [39m0.9524448[39m |


.. parsed-literal::

    | [39m27       [39m | [39m0.2051465[39m | [39m0.3692185[39m | [39m-0.191760[39m | [39m-0.075047[39m |
    | [39m28       [39m | [39m0.4125911[39m | [39m0.6886774[39m | [39m-0.046218[39m | [39m0.0114139[39m |


.. parsed-literal::

    | [35m29       [39m | [35m0.5436281[39m | [35m0.4663919[39m | [35m-0.086370[39m | [35m0.0246949[39m |
    | [35m30       [39m | [35m0.6413501[39m | [35m0.4979543[39m | [35m-0.041705[39m | [35m0.1428720[39m |


.. parsed-literal::

    | [35m31       [39m | [35m0.7819337[39m | [35m0.5710362[39m | [35m-0.045724[39m | [35m0.2137782[39m |


.. parsed-literal::

    | [39m32       [39m | [39m0.4767379[39m | [39m0.5352728[39m | [39m0.0299197[39m | [39m0.2624907[39m |


.. parsed-literal::

    | [39m33       [39m | [39m0.3013880[39m | [39m0.5579259[39m | [39m-0.130762[39m | [39m0.2397892[39m |
    | [39m34       [39m | [39m0.0138074[39m | [39m0.3362308[39m | [39m-0.840353[39m | [39m-0.837670[39m |
    | [39m35       [39m | [39m0.0013310[39m | [39m-0.483655[39m | [39m0.3866039[39m | [39m0.6740750[39m |
    | [39m36       [39m | [39m0.1174572[39m | [39m-0.049565[39m | [39m-0.108954[39m | [39m-0.202153[39m |


.. parsed-literal::

    | [39m37       [39m | [39m0.5255571[39m | [39m0.6097817[39m | [39m-0.004734[39m | [39m0.1679875[39m |


.. parsed-literal::

    | [39m38       [39m | [39m0.7379100[39m | [39m0.5218613[39m | [39m-0.034616[39m | [39m0.2090108[39m |
    | [35m39       [39m | [35m0.8490859[39m | [35m0.6323366[39m | [35m-0.032667[39m | [35m0.2707394[39m |


.. parsed-literal::

    | [39m40       [39m | [39m-.044e-07[39m | [39m-0.831439[39m | [39m0.3584873[39m | [39m-0.030378[39m |
    | [35m41       [39m | [35m0.9025055[39m | [35m0.7107871[39m | [35m-0.024897[39m | [35m0.3139254[39m |


.. parsed-literal::

    | [39m42       [39m | [39m0.8603514[39m | [39m0.6672859[39m | [39m-0.024387[39m | [39m0.3857774[39m |


.. parsed-literal::

    | [39m43       [39m | [39m0.5009260[39m | [39m0.7225206[39m | [39m0.0629590[39m | [39m0.3653689[39m |
    | [39m44       [39m | [39m0.0008535[39m | [39m-0.324396[39m | [39m0.8964272[39m | [39m-0.695403[39m |
    | [39m45       [39m | [39m0.7708845[39m | [39m0.9743153[39m | [39m0.0468998[39m | [39m0.6726593[39m |


.. parsed-literal::

    | [39m46       [39m | [39m0.2796883[39m | [39m0.7344644[39m | [39m-0.104430[39m | [39m0.3734581[39m |
    | [39m47       [39m | [39m0.5285330[39m | [39m0.4842338[39m | [39m-0.073633[39m | [39m0.0276697[39m |
    | [39m48       [39m | [39m0.1170918[39m | [39m-0.455283[39m | [39m0.0556825[39m | [39m-0.294502[39m |


.. parsed-literal::

    | [39m49       [39m | [39m0.0005530[39m | [39m-0.896720[39m | [39m-0.199863[39m | [39m-0.487141[39m |


.. parsed-literal::

    | [39m50       [39m | [39m0.6889618[39m | [39m1.0      [39m | [39m0.0906087[39m | [39m0.7673260[39m |


.. parsed-literal::

    | [39m51       [39m | [39m0.6838936[39m | [39m0.8762485[39m | [39m0.0603446[39m | [39m0.7296905[39m |
    | [39m52       [39m | [39m0.0067735[39m | [39m0.9691647[39m | [39m-0.042159[39m | [39m0.7514141[39m |


.. parsed-literal::

    | [39m53       [39m | [39m0.3685365[39m | [39m0.9467601[39m | [39m0.1473334[39m | [39m0.6715911[39m |


.. parsed-literal::

    | [39m54       [39m | [39m0.8365565[39m | [39m0.6495542[39m | [39m0.0045080[39m | [39m0.3293315[39m |
    | [39m55       [39m | [39m0.8216861[39m | [39m0.7354916[39m | [39m-0.020858[39m | [39m0.2344069[39m |


.. parsed-literal::

    | [39m56       [39m | [39m-.577e-05[39m | [39m-0.879484[39m | [39m-0.954218[39m | [39m0.7154813[39m |
    | [39m57       [39m | [39m0.7832472[39m | [39m0.5852044[39m | [39m-0.015138[39m | [39m0.4526613[39m |


.. parsed-literal::

    | [39m58       [39m | [39m0.7938804[39m | [39m0.6597925[39m | [39m0.0105868[39m | [39m0.5419519[39m |


.. parsed-literal::

    | [39m59       [39m | [39m0.2440359[39m | [39m0.5661378[39m | [39m-0.041364[39m | [39m0.5962431[39m |


.. parsed-literal::

    | [39m60       [39m | [39m0.8320164[39m | [39m0.7867958[39m | [39m0.0324158[39m | [39m0.5959492[39m |
    | [39m61       [39m | [39m0.4959325[39m | [39m0.7079475[39m | [39m0.1158142[39m | [39m0.5953950[39m |


.. parsed-literal::

    | [39m62       [39m | [39m0.1669744[39m | [39m0.1229802[39m | [39m0.1805488[39m | [39m0.0361662[39m |


.. parsed-literal::

    | [39m63       [39m | [39m0.8216539[39m | [39m0.9007799[39m | [39m0.0125311[39m | [39m0.5672909[39m |
    | [39m64       [39m | [39m0.8316071[39m | [39m0.7870544[39m | [39m0.0327162[39m | [39m0.5968212[39m |
    | [39m65       [39m | [39m0.0003834[39m | [39m-0.712240[39m | [39m0.6268556[39m | [39m0.2073338[39m |


.. parsed-literal::

    | [39m66       [39m | [39m0.0001571[39m | [39m0.5744876[39m | [39m0.9758150[39m | [39m0.6370951[39m |
    | [39m67       [39m | [39m0.0003994[39m | [39m0.3575027[39m | [39m-0.296749[39m | [39m-0.023575[39m |


.. parsed-literal::

    | [39m68       [39m | [39m0.0840023[39m | [39m0.7956128[39m | [39m-0.065188[39m | [39m0.6093132[39m |


.. parsed-literal::

    | [39m69       [39m | [39m0.6736730[39m | [39m0.8465738[39m | [39m0.0872708[39m | [39m0.5378365[39m |
    | [35m70       [39m | [35m0.9500097[39m | [35m1.0      [39m | [35m0.0343604[39m | [35m0.5404349[39m |


.. parsed-literal::

    | [35m71       [39m | [35m0.9937998[39m | [35m1.0      [39m | [35m0.0120896[39m | [35m0.4337827[39m |


.. parsed-literal::

    | [39m72       [39m | [39m0.2720796[39m | [39m1.0      [39m | [39m0.1068704[39m | [39m0.4237261[39m |


.. parsed-literal::

    | [39m73       [39m | [39m0.4196244[39m | [39m1.0      [39m | [39m-0.060757[39m | [39m0.4818069[39m |


.. parsed-literal::

    | [39m74       [39m | [39m0.9787179[39m | [39m0.9329835[39m | [39m0.0194547[39m | [39m0.4620656[39m |


.. parsed-literal::

    | [39m75       [39m | [39m0.9821017[39m | [39m0.9462511[39m | [39m-0.011583[39m | [39m0.3572611[39m |


.. parsed-literal::

    | [39m76       [39m | [39m0.8554799[39m | [39m0.8962685[39m | [39m-0.014639[39m | [39m0.2455088[39m |
    | [39m77       [39m | [39m0.1200786[39m | [39m0.4775626[39m | [39m0.1868229[39m | [39m0.1878511[39m |


.. parsed-literal::

    | [39m78       [39m | [39m0.9185996[39m | [39m1.0      [39m | [39m-0.047368[39m | [39m0.2737846[39m |


.. parsed-literal::

    | [39m79       [39m | [39m0.4297086[39m | [39m1.0      [39m | [39m0.0093534[39m | [39m0.1550549[39m |


.. parsed-literal::

    | [39m80       [39m | [39m0.4254756[39m | [39m0.9266481[39m | [39m-0.114392[39m | [39m0.2660853[39m |


.. parsed-literal::

    | [39m81       [39m | [39m0.7669268[39m | [39m1.0      [39m | [39m0.0218862[39m | [39m0.3068369[39m |


.. parsed-literal::

    | [39m82       [39m | [39m0.8139711[39m | [39m0.8550331[39m | [39m0.0184429[39m | [39m0.3242011[39m |
    | [39m83       [39m | [39m0.3778611[39m | [39m0.9226054[39m | [39m0.1628879[39m | [39m0.9119623[39m |


.. parsed-literal::

    | [39m84       [39m | [39m0.0002142[39m | [39m-0.873608[39m | [39m-0.590417[39m | [39m-0.322908[39m |
    | [39m85       [39m | [39m-.995e-06[39m | [39m0.0889854[39m | [39m-1.0     [39m | [39m1.0      [39m |


.. parsed-literal::

    | [39m86       [39m | [39m0.0547662[39m | [39m0.8283655[39m | [39m0.0747172[39m | [39m0.1985934[39m |
    | [39m87       [39m | [39m0.0      [39m | [39m-1.0     [39m | [39m1.0      [39m | [39m1.0      [39m |


.. parsed-literal::

    | [39m88       [39m | [39m0.6663736[39m | [39m0.6280386[39m | [39m0.0664455[39m | [39m0.4674775[39m |
    | [39m89       [39m | [39m0.0007961[39m | [39m1.0      [39m | [39m-1.0     [39m | [39m-0.258467[39m |


.. parsed-literal::

    | [39m90       [39m | [39m0.6252212[39m | [39m1.0      [39m | [39m-0.101559[39m | [39m-1.0     [39m |


.. parsed-literal::

    | [39m91       [39m | [39m0.0018788[39m | [39m1.0      [39m | [39m-0.274042[39m | [39m-1.0     [39m |
    | [39m92       [39m | [39m0.0009246[39m | [39m-0.723588[39m | [39m0.4703690[39m | [39m-0.997180[39m |


.. parsed-literal::

    | [39m93       [39m | [39m0.0223863[39m | [39m1.0      [39m | [39m0.0519901[39m | [39m-1.0     [39m |
    | [39m94       [39m | [39m0.9625406[39m | [39m0.8660557[39m | [39m-0.003959[39m | [39m0.4206908[39m |


.. parsed-literal::

    | [39m95       [39m | [39m0.9176641[39m | [39m0.7439024[39m | [39m0.0140623[39m | [39m0.4802418[39m |
    | [39m96       [39m | [39m0.0001722[39m | [39m1.0      [39m | [39m1.0      [39m | [39m-0.071982[39m |


.. parsed-literal::

    | [39m97       [39m | [39m0.0001046[39m | [39m-0.012447[39m | [39m-1.0     [39m | [39m0.1514002[39m |
    | [39m98       [39m | [39m0.0001749[39m | [39m-0.765238[39m | [39m-0.898005[39m | [39m0.5169565[39m |
    | [39m99       [39m | [39m0.0166398[39m | [39m-0.227062[39m | [39m-0.174050[39m | [39m0.2779688[39m |


.. parsed-literal::

    | [39m100      [39m | [39m0.5763829[39m | [39m0.7266898[39m | [39m0.1043722[39m | [39m0.8172706[39m |


.. parsed-literal::

    | [39m101      [39m | [39m0.3153274[39m | [39m0.5996019[39m | [39m0.1592724[39m | [39m0.9868877[39m |
    | [39m102      [39m | [39m0.6040294[39m | [39m0.7812510[39m | [39m-0.106681[39m | [39m-1.0     [39m |


.. parsed-literal::

    | [39m103      [39m | [39m0.6332109[39m | [39m0.8467796[39m | [39m-0.106571[39m | [39m-0.832928[39m |


.. parsed-literal::

    | [39m104      [39m | [39m0.6295450[39m | [39m0.6685843[39m | [39m-0.103822[39m | [39m-0.847980[39m |


.. parsed-literal::

    | [39m105      [39m | [39m0.0002737[39m | [39m0.7300373[39m | [39m-0.253650[39m | [39m-0.852260[39m |


.. parsed-literal::

    | [39m106      [39m | [39m0.2067497[39m | [39m0.7366109[39m | [39m0.0267531[39m | [39m-0.865456[39m |


.. parsed-literal::

    | [39m107      [39m | [39m0.5769434[39m | [39m0.5569218[39m | [39m-0.092325[39m | [39m-0.978868[39m |
    | [39m108      [39m | [39m-.780e-05[39m | [39m-0.454330[39m | [39m0.6388374[39m | [39m-0.363639[39m |


.. parsed-literal::

    | [39m109      [39m | [39m0.6671670[39m | [39m0.5032939[39m | [39m-0.080969[39m | [39m-0.800404[39m |


.. parsed-literal::

    | [39m110      [39m | [39m0.4690523[39m | [39m0.3717895[39m | [39m-0.119981[39m | [39m-0.899121[39m |
    =============================================================




.. parsed-literal::

    {'status': 0, 'value': 0.006200159446014486, 'iterations': 110}



The plot shows the all the samples that the optimization took in the
two-dimensional parameter space. The red dot marks the best value.

.. code:: ipython3

    plt.figure()
    plt.scatter([s[0] for s in samples[:-1]], [s[1] for s in samples[:-1]], c="blue")
    plt.scatter([params[0].get_value()], [params[2].get_value()], c="red", marker="o", s=100)
    plt.xlim(params[0].get_min_value()[0], params[0].get_max_value()[0])
    plt.ylim(params[2].get_min_value()[0], params[2].get_max_value()[0])
    plt.xlabel(params[0].get_name())
    plt.ylabel(params[2].get_name())
    plt.show()



.. image:: 02C_Qubit-bayesian-optimization_files/02C_Qubit-bayesian-optimization_20_0.png


.. code:: ipython3

    plot_time_evolution()




.. parsed-literal::

    (<Figure size 400x400 with 2 Axes>,
     array([<Axes: ylabel='Field [MHz]'>,
            <Axes: xlabel='Time [ns]', ylabel='Population'>], dtype=object))




.. image:: 02C_Qubit-bayesian-optimization_files/02C_Qubit-bayesian-optimization_21_1.png


We can see from the plot and optimizer output that we have found good
controls.

.. code:: ipython3

    gate_fid.measure()




.. parsed-literal::

    0.9937998405539855


