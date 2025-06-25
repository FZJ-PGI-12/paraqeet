""" """

import jax.numpy as jnp
from cthree.model.hamiltonian import Hamiltonian
from cthree.quantity import Array, Quantity
from cthree.signal.generator import Generator


class CustomHamiltonian(Hamiltonian):
    __drift_hamiltonian: Array
    __drive: list
    __generator: Generator

    def __init__(
        self,
        drift_hamiltonian: Array,
        drive: Array,
        generator: Generator,
    ):
        self.__drift_hamiltonian = drift_hamiltonian
        self.__couplings = couplings
        self.__drive = drive
        self.__generator = generator

    def get_matrix(self, t: Array) -> Array:
        """
        Return the matrix representation of the Hamiltonian.
        NOTE - Works ONLY with MultiGenerator (from customGenerator)

        Args:
            t (Array): Vector of time samples

        Returns:
            Array: Hamiltonian of shape [t, n, n]  with t: time, n: hilbert space
        """
        sig = self.__generator.generate_signal(t)
        drive = jnp.array(self.__drive)
        return jnp.sum(jnp.array(self.__drift_hamiltonian), axis=0) + jnp.sum(
            jnp.reshape(sig, sig.shape + (1, 1)) * jnp.expand_dims(drives, axis=1), axis=0
        )

    def getParameters(self) -> list[Quantity]:
        return []

    def gradient(self, t: Array) -> Array:
        """
        Return the gradient of each parameter as a list.
        NOTE - Works ONLY with MultiGenerator (from customGenerator)
        """
        drives = jnp.array(self.__drive)
        sig_grads = self.__generator.generate_signal_gradient(t)
        grads = []
        for i, sig_grad in enumerate(sig_grads):
            ham_grad = jnp.reshape(sig_grad, sig_grad.shape + (1, 1)) * drives[i]
            grads.append(ham_grad)

        all_grads = jnp.empty(shape=t.shape + (0,) + drives[0].shape)
        for grad in grads:
            all_grads = jnp.append(all_grads, grad, axis=1)
        return all_grads
