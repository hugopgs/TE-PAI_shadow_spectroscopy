# Written by: Hugo PAGES
# Date: 2024-01-05

# Standard library imports
from dataclasses import dataclass
from itertools import combinations, product
from typing import Callable, List, Tuple
import math as math

# Third-party imports
import numpy as np
from scipy import integrate
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from itertools import product


# Local application imports
from .Hamiltonian import Hamiltonian


class Heisenberg_Hamil(Hamiltonian):
    """
    Heisenberg_Hamil is a subclass of Hamiltonian that represents a time-independent 
    Heisenberg spin chain Hamiltonian for `n` qubits.

    This model includes interactions of the form:
        Jx * XX + Jy * YY + Jz * ZZ
    between neighboring qubits, with the option to enable periodic boundary conditions.

    Parameters:
    - n (int): Number of qubits (spins) in the chain.
    - jx, jy, jz (float): Coupling constants for the XX, YY, and ZZ terms.
    - boundarie_conditions (bool): If True, periodic boundary conditions are used;
    otherwise, open boundary conditions apply.
    The coefficients are constant in time but structured as functions to remain compatible 
    with the parent Hamiltonian class.
    """
    def __init__(self, n: int, jx: float, jy: float, jz: float, boundarie_conditions: bool = False, fully_connected : bool =False):
        self.jx=jx
        self.jy=jy
        self.jz=jz
        self.fully_connected=fully_connected
        def Jx(t):
            return jx

        def Jy(t):
            return jy

        def Jz(t):
            return jz
        if boundarie_conditions:
            terms = [
                (gate, [k, (k + 1) % n], Jx if gate ==
                 "XX" else Jy if gate == "YY" else Jz)
                for k, gate in product(range(n), ["XX", "YY", "ZZ"])
            ]
        else:
            terms = [
                (gate, [k, k + 1], Jx if gate ==
                 "XX" else Jy if gate == "YY" else Jz)
                for k, gate in product(range(n-1), ["XX", "YY", "ZZ"])
            ]
        if fully_connected:
            terms = [
                (gate, [i, j], Jx if gate == "XX" else Jy if gate == "YY" else Jz)
                 for (i, j), gate in product(combinations(range(n), 2), ["XX", "YY", "ZZ"])
                ]    
            
        self.name = f"Heisenberg_Jx{jx}_Jy{jy}_Jz{jz}_nq{n}"
        super().__init__(n, terms)

        
    def get_trotter_steps_from_depth(self, depth: int) -> int:
        """
        Compute the number of Trotter steps based on the desired circuit depth.

        Args:
            depth (int): Desired circuit depth.

        Returns:
            int: Number of Trotter steps.
            
        """
        if self.fully_connected:
            N_trot=int((depth -(2.8*self.nqubits-7.2))/(3.3*self.nqubits-1.7))
        else:
            N_trot=int((depth -(3*self.nqubits-7))/8)
        return N_trot