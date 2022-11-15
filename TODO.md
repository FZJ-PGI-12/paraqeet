## TODO

- fix the quantity class
- Better names
  - rename "Optimisable" into something more fitting ("QuantityProvider"?)
  - rename "Model" vs "Hamiltonian" (to Representation, even worse)
  - "SignalGenerator", "LieGroupGenerator"
- set up CI pipeline?
- replace tensorflow with jax?
- precommit
- data type that carrier values and timestamps between layers; different grid sizes within the timestamp vector should
  be possible
- find a way to pass down a time vector; maybe add a `set_time` function to all layers
- basis transformations should happen in the model
- Subspace projection:
  - projections happen in the measurement
  - the model class provides the list of states that is necessary for the projection
- how to gate sequences?
  - some fidelity classes can handle gate sets
  - they will have a full pipeline (from generator to propagation) for each gate
- examples for:
  - obtaining the signal from the generator and and plotting it
  - obtaining the propagator and plotting it
