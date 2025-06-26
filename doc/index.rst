===================================================================================
paraQeet -  A quantum optimal control toolkit with parameter handling
===================================================================================

You can find quick information on :ref:`installation <install>` and contributing in the `README`_ and `CONTRIBUTING`_ documents.

* :math:`C_1` Open-loop optimal control: Given a model, find the pulse shapes which maximize fidelity with a target operation.
* :math:`C_2`  Closed-loop calibration: Given pulses, calibrate their parameters to maximize a figure of merit measured by the actual experiment, thus improving beyond the limits of a deficient model.
* :math:`C_3`  Model learning: Given control pulses and their experimental measurement outcome, optimize model parameters to best reproduce the results.

When combined in sequence, these three procedures represent a recipe for system characterization.

*Note: This documentation is work-in-progress.*

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Examples:
   :glob:
   
   notebooks/*

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API:
   :glob:
   
   source/*


Contents
========
.. toctree::
   
   usage



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
