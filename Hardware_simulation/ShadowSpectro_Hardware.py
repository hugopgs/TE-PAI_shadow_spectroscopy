# Written by: Hugo PAGES
# Date: 2024-01-05

# Standard library imports
from itertools import chain
from typing import Union

# Third-party imports
import numpy as np
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate


class ShadowSpectro_Hardware():
    """
    Class for performing Shadow Spectroscopy on real quantum hardware.

    This class interfaces a shadow spectroscopy routine with a quantum backend
    (either real or simulated) to generate, execute, and analyze time-evolved
    quantum circuits for spectral analysis of a given Hamiltonian.
    
    Attributes:
        hardware: An object managing backend access and job submission (custom `Hardware` class).
        shadow_spectro: An instance of a shadow spectroscopy class.
        shadow: The classical shadow module used for circuit modifications and reconstruction.
        shadow_size: Number of classical shadows used at each time step.
        nq: Number of qubits in the system.
    """
    def __init__(self, hardware, shadow_spectro):
        """
        Initialize the ShadowSpectro_Hardware class.
        
        Args:
            hardware: A hardware interface class responsible for transpilation and job management.
            shadow_spectro: An instance of a shadow spectroscopy class containing parameters and methods.
        """
        self.hardware = hardware
        self.shadow_spectro = shadow_spectro
        self.shadow = shadow_spectro.shadow
        self.shadow_size = self.shadow_spectro.shadow_size
        self.nq = self.shadow_spectro.nq
        pass

    def get_time_evolve_circuits(self, hamil, init_state: Union[np.ndarray, list, QuantumCircuit] = None, verbose: bool = True, backend=None,  N_trotter_step: int = 1000):
        """
        Generate time-evolution circuits with random Clifford layers for classical shadows.

        Args:
            hamil: A function or object that returns a UnitaryGate or generates a circuit via `.gen_quantum_circuit(t)`.
            init_state: Initial state as a state vector, list, or QuantumCircuit.
            verbose: Whether to show a progress bar.
            backend: Optional backend used for Clifford randomization.
            N_trotter_step: Number of Trotter steps if Trotterization is used.

        Returns:
            Tuple:
                - List of QuantumCircuits to execute on hardware.
                - List of corresponding Clifford gates applied at each step.
        """
        
        T = np.linspace(0, self.shadow_spectro.Nt *
                        self.shadow_spectro.dt, self.shadow_spectro.Nt)
        Time_evolve_circuit_array = []
        Time_evolve_Clifford_array = []
        try:
            if isinstance(hamil(1), UnitaryGate):
                flag = True
        except:
            flag = False

        for t in tqdm(T, desc="Generate circuit Time evolution", disable=not verbose):
            if flag:
                circ = QuantumCircuit(self.nq)
                if isinstance(init_state, (np.ndarray, list)):
                    circ.initialize(init_state, normalize=True)
                circ.append(hamil(t), [n for n in range(self.nq)])
                C = circ.copy()
                if isinstance(init_state, QuantumCircuit):
                    C = init_state.compose(circ, [i for i in range(self.nq)])
            else:
                C = hamil.gen_quantum_circuit(
                    t, init_state=init_state,  N_trotter_step=N_trotter_step)

            Clifford_Gate_array = []
            for i in range(self.shadow_size):
                clifford_gate, circuit = self.shadow.add_random_clifford(
                    C, copy=True, backend=backend)
                Clifford_Gate_array.append(clifford_gate)
                Time_evolve_circuit_array.append(circuit)

            Time_evolve_Clifford_array.append(Clifford_Gate_array)

        return Time_evolve_circuit_array, Time_evolve_Clifford_array

    def send_sampler_pub(self, pubs):
        """
        Submit circuits to hardware sampler and retrieve or return job ID.
        Args:
            pubs: A list of circuits or circuits formatted for submission.

        Returns:
            List of results if using a fake backend, otherwise a job ID (or list of job IDs).
        """
        if self.hardware.Fake_backend:
            return self.hardware.get_data_from_results(self.hardware.send_sampler_pub(pubs))
        else:
            job_id = self.hardware.send_sampler_pub(pubs)
            return job_id

    def deflatten_Time_evolve_bit_string_array(self, flattenTime_evolve_bit_string_array):
        """
        Restructure flat measurement results into time-evolution organized structure.

        Args:
            flattenTime_evolve_bit_string_array: A flat list of measurement results.

        Returns:
            Nested list: [Nt][shadow_size] formatted bitstrings.
        """
        Time_evolve_bit_string_array = []
        for nt in range(self.shadow_spectro.Nt):
            tmp = []
            for shadow in range(self.shadow_size):
                tmp.append(flattenTime_evolve_bit_string_array[0])
                flattenTime_evolve_bit_string_array.pop(0)
            Time_evolve_bit_string_array.append(tmp)
        return Time_evolve_bit_string_array

    def get_data_matrix_from_hardware(self, job_id: str, Clifford: list[list[UnitaryGate]], density_matrix: bool = True, verbose: bool = False, res_fake_backend: list = None):
        """
        Reconstruct the data matrix from hardware results, using classical shadows.

        Args:
            job_id: Job identifier(s) returned from hardware submission.
            Clifford: Nested list of Clifford gates used in measurements.
            density_matrix: If True, reconstruct the density matrix at each step.
            verbose: Whether to print progress.
            res_fake_backend: Optional list of precomputed bitstrings (used in simulation mode).

        Returns:
            Tuple:
                - Reconstructed spectral signal (array).
                - Extracted frequency components (array).
        """
        
        Measurement_bit_string = []
        if res_fake_backend is not None:
            Measurement_bit_string = res_fake_backend
        else:
            raw_data = []
            for id in job_id:
                raw_data.append(self.hardware.get_sampler_result(id))

            for sublist in raw_data:
                for bits in sublist:
                    Measurement_bit_string.append(bits)
        Bit_string_array = self.deflatten_Time_evolve_bit_string_array(
            Measurement_bit_string)
        D = []
        for t in tqdm(range(self.shadow_spectro.Nt), desc="Spectral cross_correlation", disable=not verbose):
            if density_matrix:
                Rho = np.zeros((2**self.nq, 2**self.nq), dtype='complex128')
                for i in range(self.shadow_size):
                    Rho += self.shadow.snapshot_density_matrix(
                        Clifford[t][i], Bit_string_array[t][i])
                Rho = Rho/self.shadow_size
                fkt = self.shadow_spectro.expectation_value_q_Pauli(
                    0, 0, density_matrix=Rho)
                D.append(fkt.tolist())
            else:
                fkt = self.shadow_spectro.expectation_value_q_Pauli(
                    Clifford[t], Bit_string_array[t])
                D.append(fkt.tolist())
        D = np.array(D).real

        solution, frequencies = self.shadow_spectro.spectro.Spectroscopy(D)

        return solution, frequencies
