Single spin: Bayesian optimization of a gate
============================================

This is similar to the 02B_Single_qubit_gate example, except that it
uses Bayesian optimization (Shahriari et al., 2016)
:cite:p:`shahriari2016taking` instead of gradient descent.

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    from jax import Array
    
    from paraqeet.eom.schroedinger_equation import SchroedingerEquation
    from paraqeet.hamiltonian.drive import Drive
    from paraqeet.hamiltonian.qubit import QubitHamiltonian
    from paraqeet.logger import Logger
    from paraqeet.measurement.unitary_fidelity import UnitaryFidelity
    from paraqeet.optimization_map import OptimizationMap
    from paraqeet.optimizers.bayesian_optimizer import BayesianOptimizer
    from paraqeet.propagation import Expm
    from paraqeet.quantity import Quantity
    from paraqeet.signal.envelopes import ConstantEnvelope
    from paraqeet.signal.iq_mixer import IQMixer

Setup
-----

We first set up the qubit system we want to control. We set the qubit
frequency :math:`\omega_q / 2 \pi` to be :math:`4.8` GHz and define the
Hamiltonian as

.. math:: H(t)=H_\text{drift}+H_c(t)= \frac{\omega_q}{2} \sigma_z + \Omega(t)\sigma_x, 

\ where :math:`\Omega(t)` will be supplied by the generator.

.. code:: ipython3

    freq_q = 4.8e9
    omega_q = 2 * np.pi * freq_q
    
    qubit_hamiltonian = QubitHamiltonian(
        frequency=Quantity(omega_q, 0.8 * omega_q, 1.2 * omega_q, unit="Hz", two_pi=True), drives=[]
    )
    model = SchroedingerEquation(
        hamiltonian_func=qubit_hamiltonian.get_value,
        hamiltonian_gradient_func=qubit_hamiltonian.get_gradient,
    )

For signal generation, we define a simple cosine shaped tone generator
:math:`A \cos(\omega t)`

.. code:: ipython3

    t_simu = 3e-9
    tone = ConstantEnvelope()
    tone.t_final.set_value(t_simu)
    gen = IQMixer(envelopes=[tone])

We can inspect the pre-defined parameters with

.. code:: ipython3

    params_gen = gen.get_parameters()

In this notebook, we would like to optimize the amplitude ``Amplitude``
and frequency ``lo_freq`` of the drive. We add a drive on the qubit.

.. code:: ipython3

    freq = 4.8e9 * 2 * np.pi
    sigma_x = qubit_hamiltonian.sigma_x
    drive = Drive(sigma_x, gen)
    qubit_hamiltonian.drives = [drive]
    model = SchroedingerEquation(
        hamiltonian_func=qubit_hamiltonian.get_value,
        hamiltonian_gradient_func=qubit_hamiltonian.get_gradient,
    )

Textbook values for implementing an :math:`X` rotation on this system at
a time :math:`T` would be :math:`\omega=\omega_q` and :math:`A=\pi/T`.
We use some offset from these values as an initial guess to demonstrate
the optimization procedure.

.. code:: ipython3

    params_gen[0].set_value(0.5 * np.pi / t_simu)
    params_gen[2].set_value(1.01 * freq)

We select a propagation method, piecewise constant exponentiation, and
configure an :math:`X`-gate as a target gate. Also, we initialize the
identity at time :math:`0`.

.. code:: ipython3

    times = np.array([0.0, t_simu])
    
    prop = Expm(eom_func=model.get_value, resolution=100e9, initial_state=np.identity(2))
    gate_fid = UnitaryFidelity(propagation_func=prop.get_value, propagation_gradient_func=None, gate=sigma_x)

.. code:: ipython3

    from plotting import plot_signal_and_dynamics
    
    ts = np.linspace(0.0, t_simu, 301)
    plot_signal_and_dynamics(gen, prop, ts, state_labels=[r"$|0\rangle$", r"$|1\rangle$"])




.. parsed-literal::

    array([<Axes: ylabel='Amplitude \n[MHz / $2\\pi$]'>,
           <Axes: xlabel='Time [ns]', ylabel='Population'>], dtype=object)




.. image:: 02C_Qubit-bayesian-optimization_files/02C_Qubit-bayesian-optimization_14_1.png


As expected, we get a partial transfer and a low fidelity.

.. code:: ipython3

    print(f"Gate fidelity: {gate_fid.get_value(times)}")


.. parsed-literal::

    Gate fidelity: 0.008842872531522971


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
    optmap.add(gen, [params_gen[0], params_gen[2], params_gen[3]])
    opt = BayesianOptimizer(measure_func=gate_fid.get_value, optimization_map=optmap, initial_samples=10, iterations=100)
    opt.logger = CustomLogger()

.. code:: ipython3

    opt.optimize(times)


.. parsed-literal::

    |   iter    |  target   |     0     |     1     |     2     |
    -------------------------------------------------------------
    | [39m1        [39m | [39m0.0036278[39m | [39m-0.165955[39m | [39m0.4406489[39m | [39m-0.999771[39m |
    | [39m2        [39m | [39m0.0001104[39m | [39m-0.395334[39m | [39m-0.706488[39m | [39m-0.815322[39m |
    | [39m3        [39m | [39m0.0003939[39m | [39m-0.627479[39m | [39m-0.308878[39m | [39m-0.206465[39m |
    | [35m4        [39m | [35m0.0094778[39m | [35m0.0776334[39m | [35m-0.161610[39m | [35m0.3704390[39m |
    | [39m5        [39m | [39m-.085e-06[39m | [39m-0.591095[39m | [39m0.7562348[39m | [39m-0.945224[39m |
    | [35m6        [39m | [35m0.2028943[39m | [35m0.3409350[39m | [35m-0.165390[39m | [35m0.1173796[39m |
    | [39m7        [39m | [39m0.0007805[39m | [39m-0.719226[39m | [39m-0.603797[39m | [39m0.6014891[39m |
    | [39m8        [39m | [39m0.0434133[39m | [39m0.9365231[39m | [39m-0.373151[39m | [39m0.3846452[39m |
    | [39m9        [39m | [39m-.729e-06[39m | [39m0.7527783[39m | [39m0.7892133[39m | [39m-0.829911[39m |
    | [39m10       [39m | [39m-.574e-06[39m | [39m-0.921890[39m | [39m-0.660339[39m | [39m0.7562850[39m |


.. parsed-literal::

    | [35m11       [39m | [35m0.2425745[39m | [35m0.3864646[39m | [35m-0.166284[39m | [35m0.0747446[39m |


.. parsed-literal::

    | [35m12       [39m | [35m0.3210977[39m | [35m0.3995182[39m | [35m-0.075493[39m | [35m-0.066396[39m |
    | [39m13       [39m | [39m0.1647017[39m | [39m0.5639544[39m | [39m-0.194260[39m | [39m-0.341354[39m |
    | [35m14       [39m | [35m0.3215744[39m | [35m0.5552035[39m | [35m0.1456268[39m | [35m-0.017306[39m |


.. parsed-literal::

    | [39m15       [39m | [39m-.246e-06[39m | [39m0.2755778[39m | [39m0.3197374[39m | [39m-0.141076[39m |
    | [39m16       [39m | [39m0.2212795[39m | [39m0.5554470[39m | [39m0.1939444[39m | [39m-0.022655[39m |
    | [39m17       [39m | [39m0.1641519[39m | [39m0.5808638[39m | [39m-0.020411[39m | [39m-0.013947[39m |


.. parsed-literal::

    | [35m18       [39m | [35m0.3386069[39m | [35m0.3983636[39m | [35m-0.078108[39m | [35m-0.062977[39m |


.. parsed-literal::

    | [35m19       [39m | [35m0.4735536[39m | [35m0.3644688[39m | [35m-0.102168[39m | [35m-0.005663[39m |
    | [39m20       [39m | [39m0.3445419[39m | [39m0.3365286[39m | [39m-0.152913[39m | [39m-0.047864[39m |


.. parsed-literal::

    | [39m21       [39m | [39m0.2996809[39m | [39m0.3799188[39m | [39m-0.030457[39m | [39m0.0350806[39m |
    | [39m22       [39m | [39m0.3506612[39m | [39m0.2964954[39m | [39m-0.068823[39m | [39m-0.020524[39m |
    | [39m23       [39m | [39m0.0011528[39m | [39m0.9867827[39m | [39m-0.644633[39m | [39m0.8585118[39m |
    | [39m24       [39m | [39m-.894e-05[39m | [39m-0.472166[39m | [39m-0.869049[39m | [39m-0.553717[39m |


.. parsed-literal::

    | [39m25       [39m | [39m-.132e-08[39m | [39m-0.877018[39m | [39m-0.657881[39m | [39m0.0207137[39m |
    | [39m26       [39m | [39m0.0011949[39m | [39m-0.795254[39m | [39m-0.508318[39m | [39m0.9524448[39m |


.. parsed-literal::

    | [39m27       [39m | [39m0.4317754[39m | [39m0.4324396[39m | [39m-0.137114[39m | [39m-0.016619[39m |
    | [39m28       [39m | [39m0.0843465[39m | [39m0.5621222[39m | [39m0.1172306[39m | [39m0.1098338[39m |


.. parsed-literal::

    | [39m29       [39m | [39m0.4592509[39m | [39m0.6068948[39m | [39m0.1109552[39m | [39m-0.122137[39m |


.. parsed-literal::

    | [39m30       [39m | [39m0.4500461[39m | [39m0.7011788[39m | [39m0.1225493[39m | [39m-0.096994[39m |
    | [35m31       [39m | [35m0.5526225[39m | [35m0.6880545[39m | [35m0.1237988[39m | [35m-0.212676[39m |


.. parsed-literal::

    | [39m32       [39m | [39m0.2963422[39m | [39m0.7218859[39m | [39m0.0311079[39m | [39m-0.204660[39m |
    | [39m33       [39m | [39m0.0725034[39m | [39m0.6894703[39m | [39m0.2209747[39m | [39m-0.202374[39m |


.. parsed-literal::

    | [39m34       [39m | [39m0.0137368[39m | [39m0.3362308[39m | [39m-0.840353[39m | [39m-0.837670[39m |
    | [35m35       [39m | [35m0.6050386[39m | [35m0.6234342[39m | [35m0.0913049[39m | [35m-0.226812[39m |


.. parsed-literal::

    | [39m36       [39m | [39m0.1168021[39m | [39m-0.049565[39m | [39m-0.108954[39m | [39m-0.202153[39m |
    | [35m37       [39m | [35m0.6613792[39m | [35m0.6515588[39m | [35m0.0986467[39m | [35m-0.300028[39m |


.. parsed-literal::

    | [39m38       [39m | [39m0.6427795[39m | [39m0.5720024[39m | [39m0.0964943[39m | [39m-0.330979[39m |
    | [35m39       [39m | [35m0.7138039[39m | [35m0.6342141[39m | [35m0.0810681[39m | [35m-0.410679[39m |


.. parsed-literal::

    | [39m40       [39m | [39m-.035e-07[39m | [39m-0.831439[39m | [39m0.3584873[39m | [39m-0.030378[39m |


.. parsed-literal::

    | [39m41       [39m | [39m0.1030228[39m | [39m0.6117614[39m | [39m0.1681040[39m | [39m-0.428471[39m |
    | [39m42       [39m | [39m0.6177010[39m | [39m0.6223520[39m | [39m0.0172091[39m | [39m-0.367927[39m |


.. parsed-literal::

    | [35m43       [39m | [35m0.8360415[39m | [35m0.7096159[39m | [35m0.0516334[39m | [35m-0.406770[39m |
    | [39m44       [39m | [39m0.0008451[39m | [39m-0.324396[39m | [39m0.8964272[39m | [39m-0.695403[39m |
    | [39m45       [39m | [39m0.7703155[39m | [39m0.9743153[39m | [39m0.0468998[39m | [39m0.6726593[39m |


.. parsed-literal::

    | [35m46       [39m | [35m0.8642772[39m | [35m0.7028993[39m | [35m0.0132453[39m | [35m-0.486736[39m |


.. parsed-literal::

    | [35m47       [39m | [35m0.9247866[39m | [35m0.8027911[39m | [35m0.0301315[39m | [35m-0.485857[39m |
    | [39m48       [39m | [39m0.1162966[39m | [39m-0.455283[39m | [39m0.0556825[39m | [39m-0.294502[39m |
    | [39m49       [39m | [39m0.0005492[39m | [39m-0.896720[39m | [39m-0.199863[39m | [39m-0.487141[39m |


.. parsed-literal::

    | [39m50       [39m | [39m0.2695574[39m | [39m0.8031122[39m | [39m-0.066686[39m | [39m-0.485512[39m |


.. parsed-literal::

    | [39m51       [39m | [39m0.4891204[39m | [39m0.7696168[39m | [39m0.0946999[39m | [39m-0.517352[39m |


.. parsed-literal::

    | [39m52       [39m | [39m0.8797236[39m | [39m0.8181353[39m | [39m0.0512836[39m | [39m-0.418533[39m |


.. parsed-literal::

    | [39m53       [39m | [39m0.8835306[39m | [39m0.8908719[39m | [39m0.0559286[39m | [39m-0.473369[39m |


.. parsed-literal::

    | [39m54       [39m | [39m0.5029458[39m | [39m0.9895918[39m | [39m0.1379915[39m | [39m0.7384613[39m |


.. parsed-literal::

    | [39m55       [39m | [39m0.5246147[39m | [39m0.8660061[39m | [39m0.0128609[39m | [39m0.6739320[39m |
    | [39m56       [39m | [39m-.541e-05[39m | [39m-0.879484[39m | [39m-0.954218[39m | [39m0.7154813[39m |


.. parsed-literal::

    | [39m57       [39m | [39m0.8529050[39m | [39m0.9997355[39m | [39m0.0692861[39m | [39m0.5637266[39m |


.. parsed-literal::

    | [39m58       [39m | [39m0.3471668[39m | [39m1.0      [39m | [39m-0.039209[39m | [39m0.5793447[39m |
    | [39m59       [39m | [39m0.2922228[39m | [39m0.9356930[39m | [39m0.1417109[39m | [39m0.5855986[39m |


.. parsed-literal::

    | [39m60       [39m | [39m0.7454913[39m | [39m0.9328742[39m | [39m0.0904889[39m | [39m-0.391478[39m |


.. parsed-literal::

    | [39m61       [39m | [39m0.6214151[39m | [39m1.0      [39m | [39m0.0882262[39m | [39m-0.496075[39m |
    | [39m62       [39m | [39m0.5161665[39m | [39m0.9811929[39m | [39m0.1365475[39m | [39m0.7444498[39m |


.. parsed-literal::

    | [39m63       [39m | [39m0.6789449[39m | [39m0.9501645[39m | [39m-0.002108[39m | [39m-0.416996[39m |
    | [39m64       [39m | [39m0.0028832[39m | [39m0.6475185[39m | [39m-0.866986[39m | [39m0.2989424[39m |
    | [39m65       [39m | [39m0.0003793[39m | [39m-0.712240[39m | [39m0.6268556[39m | [39m0.2073338[39m |


.. parsed-literal::

    | [39m66       [39m | [39m0.0001554[39m | [39m0.5744876[39m | [39m0.9758150[39m | [39m0.6370951[39m |
    | [39m67       [39m | [39m0.0004002[39m | [39m0.3575027[39m | [39m-0.296749[39m | [39m-0.023575[39m |


.. parsed-literal::

    | [39m68       [39m | [39m0.6433174[39m | [39m1.0      [39m | [39m0.0747461[39m | [39m0.4536843[39m |
    | [39m69       [39m | [39m0.0719186[39m | [39m0.9819331[39m | [39m-0.005330[39m | [39m0.7978891[39m |


.. parsed-literal::

    | [39m70       [39m | [39m0.7855935[39m | [39m0.5960585[39m | [39m-0.004125[39m | [39m-0.512132[39m |


.. parsed-literal::

    | [39m71       [39m | [39m0.8401320[39m | [39m0.6270568[39m | [39m-0.026322[39m | [39m-0.622177[39m |


.. parsed-literal::

    | [39m72       [39m | [39m0.6730900[39m | [39m0.5174064[39m | [39m-0.050482[39m | [39m-0.632539[39m |


.. parsed-literal::

    | [39m73       [39m | [39m0.1059564[39m | [39m0.6089376[39m | [39m-0.119924[39m | [39m-0.588921[39m |
    | [39m74       [39m | [39m0.5234386[39m | [39m0.5791422[39m | [39m0.0552139[39m | [39m-0.637371[39m |


.. parsed-literal::

    | [39m75       [39m | [39m0.8687634[39m | [39m0.7092606[39m | [39m-0.000667[39m | [39m-0.678066[39m |


.. parsed-literal::

    | [39m76       [39m | [39m0.8322264[39m | [39m0.6351141[39m | [39m-0.030434[39m | [39m-0.747049[39m |
    | [39m77       [39m | [39m0.1196956[39m | [39m0.4775626[39m | [39m0.1868229[39m | [39m0.1878511[39m |
    | [39m78       [39m | [39m0.8588845[39m | [39m0.8171016[39m | [39m0.0613024[39m | [39m-0.428494[39m |


.. parsed-literal::

    | [39m79       [39m | [39m0.7013931[39m | [39m0.7520536[39m | [39m-0.012540[39m | [39m-0.801662[39m |


.. parsed-literal::

    | [39m80       [39m | [39m0.8251733[39m | [39m0.8749484[39m | [39m0.0230114[39m | [39m-0.656802[39m |


.. parsed-literal::

    | [39m81       [39m | [39m0.0214752[39m | [39m0.8787445[39m | [39m0.0980545[39m | [39m-0.762461[39m |


.. parsed-literal::

    | [39m82       [39m | [39m0.7054977[39m | [39m0.8168963[39m | [39m-0.073347[39m | [39m-0.696622[39m |
    | [39m83       [39m | [39m0.7031685[39m | [39m0.9319049[39m | [39m0.0973267[39m | [39m-0.384528[39m |
    | [39m84       [39m | [39m0.0002128[39m | [39m-0.873608[39m | [39m-0.590417[39m | [39m-0.322908[39m |


.. parsed-literal::

    | [39m85       [39m | [39m0.0019041[39m | [39m0.5024768[39m | [39m-0.860733[39m | [39m0.3299947[39m |


.. parsed-literal::

    | [39m86       [39m | [39m0.9104421[39m | [39m0.9543916[39m | [39m-0.028352[39m | [39m-0.608714[39m |


.. parsed-literal::

    | [39m87       [39m | [39m0.5084796[39m | [39m1.0      [39m | [39m-0.097431[39m | [39m-0.697569[39m |


.. parsed-literal::

    | [39m88       [39m | [39m0.5954764[39m | [39m0.6538100[39m | [39m-0.111085[39m | [39m-0.872570[39m |
    | [39m89       [39m | [39m0.0074612[39m | [39m-0.127203[39m | [39m-0.291009[39m | [39m-0.601715[39m |


.. parsed-literal::

    | [39m90       [39m | [39m0.7212891[39m | [39m0.4793080[39m | [39m-0.047544[39m | [39m-0.801404[39m |


.. parsed-literal::

    | [39m91       [39m | [39m0.0744798[39m | [39m0.5848782[39m | [39m0.0427544[39m | [39m-0.879393[39m |
    | [39m92       [39m | [39m0.0009172[39m | [39m-0.723588[39m | [39m0.4703690[39m | [39m-0.997180[39m |


.. parsed-literal::

    | [35m93       [39m | [35m0.9739757[39m | [35m0.8810222[39m | [35m0.0100648[39m | [35m-0.571485[39m |


.. parsed-literal::

    | [39m94       [39m | [39m0.4856802[39m | [39m0.3563276[39m | [39m-0.089477[39m | [39m-0.729510[39m |


.. parsed-literal::

    | [39m95       [39m | [39m0.0776399[39m | [39m0.5006620[39m | [39m-0.185180[39m | [39m-0.817513[39m |
    | [39m96       [39m | [39m0.2675017[39m | [39m0.8040816[39m | [39m-0.159214[39m | [39m-0.860836[39m |


.. parsed-literal::

    | [39m97       [39m | [39m0.9507493[39m | [39m0.7930022[39m | [39m-0.000173[39m | [39m-0.605471[39m |
    | [39m98       [39m | [39m0.0001739[39m | [39m-0.765238[39m | [39m-0.898005[39m | [39m0.5169565[39m |
    | [39m99       [39m | [39m0.0165422[39m | [39m-0.227062[39m | [39m-0.174050[39m | [39m0.2779688[39m |


.. parsed-literal::

    | [39m100      [39m | [39m0.2512042[39m | [39m0.3942487[39m | [39m0.0544503[39m | [39m-0.736694[39m |
    | [39m101      [39m | [39m0.6713142[39m | [39m0.4207872[39m | [39m0.0082869[39m | [39m-0.461412[39m |


.. parsed-literal::

    | [39m102      [39m | [39m-.840e-06[39m | [39m-0.838311[39m | [39m0.7147163[39m | [39m-0.902185[39m |
    | [39m103      [39m | [39m0.1634979[39m | [39m0.2705901[39m | [39m-0.073235[39m | [39m-0.490517[39m |


.. parsed-literal::

    | [39m104      [39m | [39m0.8174814[39m | [39m0.8789252[39m | [39m-0.043038[39m | [39m-0.618030[39m |
    | [39m105      [39m | [39m0.7748375[39m | [39m1.0      [39m | [39m0.0417764[39m | [39m-0.621004[39m |


.. parsed-literal::

    | [39m106      [39m | [39m0.0628136[39m | [39m-0.231786[39m | [39m-0.098097[39m | [39m-0.206950[39m |
    | [39m107      [39m | [39m0.0144301[39m | [39m0.7975162[39m | [39m0.2655924[39m | [39m0.9053020[39m |


.. parsed-literal::

    | [39m108      [39m | [39m-.685e-05[39m | [39m-0.454330[39m | [39m0.6388374[39m | [39m-0.363639[39m |
    | [39m109      [39m | [39m0.0194496[39m | [39m1.0      [39m | [39m0.1289968[39m | [39m0.2124224[39m |


.. parsed-literal::

    | [39m110      [39m | [39m-.449e-06[39m | [39m0.2988408[39m | [39m-0.849121[39m | [39m0.4582089[39m |
    =============================================================




.. parsed-literal::

    {'status': 0, 'value': 0.026024268881238988, 'iterations': 110}



The plot shows all the samples that the optimization took in the
two-dimensional parameter space. The red dot marks the best value.

.. code:: ipython3

    plt.figure(figsize=(4, 4))
    plt.scatter([s[0] for s in samples[:-1]], [s[1] for s in samples[:-1]], c="blue")
    plt.scatter([params_gen[0].get_value()], [params_gen[2].get_value()], c="red", marker="o", s=100)
    plt.xlim(params_gen[0].get_min_value()[0], params_gen[0].get_max_value()[0])
    plt.ylim(params_gen[2].get_min_value()[0], params_gen[2].get_max_value()[0])
    plt.xlabel(params_gen[0].get_name())
    plt.ylabel(params_gen[2].get_name())
    plt.show()



.. image:: 02C_Qubit-bayesian-optimization_files/02C_Qubit-bayesian-optimization_21_0.png


.. code:: ipython3

    plot_signal_and_dynamics(gen, prop, ts, state_labels=[r"$|0\rangle$", r"$|1\rangle$"])




.. parsed-literal::

    array([<Axes: ylabel='Amplitude \n[MHz / $2\\pi$]'>,
           <Axes: xlabel='Time [ns]', ylabel='Population'>], dtype=object)




.. image:: 02C_Qubit-bayesian-optimization_files/02C_Qubit-bayesian-optimization_22_1.png


We can see from the plot and optimizer output that we have found good
controls.

.. code:: ipython3

    print(f"Gate fidelity: {gate_fid.get_value(times)}")


.. parsed-literal::

    Gate fidelity: 0.973975731118761


References
----------

- **(Shahriari et al., 2016)** B. Shahriari et al., “Taking the human
  out of the loop: A review of Bayesian optimization,” *Proceedings of
  the IEEE* **104**, 148–175 (2016).
