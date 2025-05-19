"""
Python Module to create and manipulate Hamiltonian that can be write as trensor product of Pauli operator 
Can be used and simulate using Qisqit 
"""

__version__="1.0.0"
__date__="13/05/2025"
__author__="Hugo PAGES"

__email__="hugo.pages@etu.unistra.fr"

    
from .Ising_hamil import Ising_Hamil
from .Heisenberg_hamil import Heisenberg_Hamil
from .Spin_chain_hamil import Spin_Chain_Hamil
from .Hamiltonian import Hamiltonian