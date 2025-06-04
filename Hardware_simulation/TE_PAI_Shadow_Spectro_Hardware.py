# Written by: Hugo PAGES
# Date: 2024-01-05

# Standard library imports
import os
import sys
import time
import gc
import pickle
import multiprocessing as mp


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import numpy as np
from tqdm import tqdm
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate, RXXGate, RYYGate, RZZGate
from qiskit.qasm2 import loads, CustomInstruction
from qiskit_ibm_runtime import SamplerV2 as Sampler, Batch

# Local application imports
from TE_PAI.TE_PAI import TE_PAI
from tools_box.data_file_functions import save_to_file


class GateConstructor:
    def __init__(self, matrix):
        self.matrix = matrix

    def __call__(self):
        return UnitaryGate(self.matrix)


class TE_PAI_Shadow_Spectro_Hardware:
    """
    Hardware interface for TE-PAI shadow spectroscopy.

    This class orchestrates time-evolution circuit generation, classical shadow sampling, and
    hardware execution for spectral analysis using the TE-PAI (Time Evolution – Projected Approximate Integration)
    algorithm and classical shadows. It supports execution on both simulated and real quantum backends.

    Attributes:
        hardware: A Hardware class instance that wraps Qiskit Runtime or simulator backends.
        te_pai_shadow_spectro: TE_PAI_Shadow_Spectro instance containing spectroscopy parameters.
        Various spectroscopy parameters (e.g., nq, Nt, dt, shadow_size, etc.).
        Clifford_Gate_set: Predefined set of Clifford strings for randomized classical shadow sampling.
        gate_set: Dictionary of basic gates (X, Y, Z, H, etc.) used to build Clifford unitaries.
        custom_instruction_list: List of `CustomInstruction` instances to support circuit deserialization.
        file_name: Filename stem for storing results and metadata.
    """
    
    def create_gate_function(self, matrix):
        return GateConstructor(matrix)

    def __init__(self, hardware, te_pai_shadow_spectro):
        self.hardware = hardware
        self.delta = te_pai_shadow_spectro.delta
        self.te_pai_shadow_spectro = te_pai_shadow_spectro
        self.shadow_spectro = te_pai_shadow_spectro.Shadow_Spectro
        self.shadow = te_pai_shadow_spectro.classical_shadow
        self.shadow_size = te_pai_shadow_spectro.shadow_size
        self.nq = te_pai_shadow_spectro.nq
        self.Nt = self.te_pai_shadow_spectro.Nt
        self.dt = self.te_pai_shadow_spectro.dt
        self.trotter_steps = te_pai_shadow_spectro.trotter_steps
        self.PAI_error = te_pai_shadow_spectro.PAI_error
        self.num_processes = min(40, int(mp.cpu_count() * 0.5))
        self.N_trotter_max = te_pai_shadow_spectro.N_trotter_max
        self.hamil = te_pai_shadow_spectro.hamil
        self.M_sample = []
        self.init_state = te_pai_shadow_spectro.init_state
        self.file_name = f"nq{self.nq}_Nt{self.Nt}_dt{self.dt:.2}_NTrot{self.N_trotter_max}_NsTE{self.shadow_size}"
        self.X = np.array([[0, 1], [1, 0]])

        self.Y = np.array([[0, -1j], [1j, 0]])
        self.Z = np.array([[1, 0], [0, -1]])
        self.I = np.array([[1, 0], [0, 1]])
        self.S = np.array([[1, 0], [0, 1j]])
        self.H = np.array(
            [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]]
        )
        self.V = self.H @ self.S @ self.H @ self.S
        self.W = self.V @ self.V
        self.gate_set = {
            "X": self.X,
            "Y": self.Y,
            "Z": self.Z,
            "I": self.I,
            "S": self.S,
            "H": self.H,
            "V": self.V,
            "W": self.W,
        }
        self.Clifford_Gate_set = [
            "III",
            "XII",
            "YII",
            "ZII",
            "VII",
            "VXI",
            "VYI",
            "VZI",
            "WXI",
            "WYI",
            "WZI",
            "HXI",
            "HYI",
            "HZI",
            "HII",
            "HVI",
            "HVX",
            "HVY",
            "HVZ",
            "HWI",
            "HWX",
            "HWY",
            "HWZ",
            "WII",
        ]
        # Extracting the gate to apply
        rxx_custom = CustomInstruction(
            name="rxx", num_params=1, num_qubits=2, builtin=False, constructor=RXXGate
        )
        ryy_custom = CustomInstruction(
            name="ryy", num_params=1, num_qubits=2, builtin=False, constructor=RYYGate
        )
        rzz_custom = CustomInstruction(
            name="rzz", num_params=1, num_qubits=2, builtin=False, constructor=RZZGate
        )
        self.custom_instruction_list = [rxx_custom, ryy_custom, rzz_custom]
        for gate in self.Clifford_Gate_set:
            gate_matrix = (
                self.gate_set[gate[0]] @ self.gate_set[gate[1]] @ self.gate_set[gate[2]]
            )
            gate_constructor = self.create_gate_function(gate_matrix)
            gate_instruction = CustomInstruction(
                name=str.lower(gate),
                num_params=0,
                num_qubits=1,
                builtin=False,
                constructor=gate_constructor,
            )
            self.custom_instruction_list.append(gate_instruction)

       
    def process_sample(self, circ):
        """Function to process a single M_sample iteration"""
        Clifford_Gate_array = []
        Time_evolve_circuit_array = []
        for _ in range(self.shadow_size):
            clifford_gates, transpiled_circ = (
                self.shadow.add_random_clifford(circ, copy= True, backend=self.hardware.backend)
            )
            # circuit_str,clifford_gates= self.shadow.add_clifford_gate_to_qasm(circ)
            # circ_copy=self.deserialize_circuit(circuit_str)
            # circ_copy.measure_all()
            # transpiled_circ = transpile(circ_copy, self.hardware.backend, optimization_level=3)
            Clifford_Gate_array.append(clifford_gates)
            Time_evolve_circuit_array.append(transpiled_circ)
        return Clifford_Gate_array, Time_evolve_circuit_array

    def deflatten_Time_evolve_bit_string_array(
        self, flattenTime_evolve_bit_string_array
    ):
        Time_evolve_bit_string_array = []
        for nt in range(self.Nt):
            tmp_circ = []
            for ms in range(self.M_sample[nt]):
                tmp_shadow = []
                for i in range(self.shadow_size):
                    tmp_shadow.append(flattenTime_evolve_bit_string_array[0])
                    flattenTime_evolve_bit_string_array.pop(0)
                tmp_circ.append(tmp_shadow)
            Time_evolve_bit_string_array.append(tmp_circ)

        return Time_evolve_bit_string_array

    def get_data_matrix_from_hardware(
        self,
        job_id: str,
        GAMMA: np.ndarray,
        Clifford: list[list[UnitaryGate]],
        is_density_matrix: bool = True,
        is_verbose: bool = False,
        res_fake_backend=None,
        Folder_path="data/Hardware_simulation/",
    ):
        Measurement_bit_string = []
        if res_fake_backend is not None:
            Measurement_bit_string=[]
            for sub_list in res_fake_backend: 
                for sub_sub_list in sub_list : 
                    for bit in sub_sub_list:
                        Measurement_bit_string.append(bit)
        else:
            raw_data = []
            for n,id in enumerate(job_id):
                for sub_id in id:
                    raw_data.append(self.hardware.get_sampler_result(sub_id))
                print("data from job id: ", n+1, "/", len(job_id))
            for sublist in raw_data:
                for bits in sublist:
                    Measurement_bit_string.append(bits)
                    
            save_to_file(
                self.file_name+"_measurement_bit_string",
                folder_name=Folder_path,  # We suppose that the file is run in Frontier_Lab, such as 'python Hardware_simulation/post_process.py'
                use_auto_structure=False,
                format="pickle",
                Measurement_bit_string=Measurement_bit_string)            
                    
        Bit_string_array = self.deflatten_Time_evolve_bit_string_array(
            Measurement_bit_string
        )

        D = []
        for nt in tqdm(
            range(self.shadow_spectro.Nt),
            desc="Spectral cross_correlation",
            disable=not is_verbose,
        ):
            if is_density_matrix:
                Rho = np.zeros((2**self.nq, 2**self.nq), dtype="complex128")
                for ms in range(self.M_sample[nt]):
                    for i in range(self.shadow_size):
                        Rho += (
                            self.te_pai_shadow_spectro.classical_shadow.snapshot_density_matrix(
                                Clifford[nt][ms][i], Bit_string_array[nt][ms][i]
                            )
                            * GAMMA[nt][ms]
                        ) / self.shadow_size
                Rho = Rho / self.M_sample[nt]
                
                fk = self.shadow_spectro.expectation_value_q_pauli(Rho)

            else:
                self.nt=nt
                expectation_values = []
                self.clifford=Clifford[nt]
                self.Bit_string_array=Bit_string_array[nt]
                self.GAMMA=GAMMA[nt]
                with mp.Pool(self.num_processes) as pool:
                    expectation_values += pool.map(self.loop_expectation_value_q_pauli,[ms for ms in range(self.M_sample[nt])] )
                fk = np.sum(expectation_values, axis=0) / self.M_sample[nt]
            fk.tolist()
            D.append(fk)

        D = np.array(D)
        save_to_file(
                self.file_name+"_Data_matrix",
                folder_name=Folder_path,  # We suppose that the file is run in Frontier_Lab, such as 'python Hardware_simulation/post_process.py'
                use_auto_structure=False,
                format="pickle",
                D=D)            
        
        solution, frequencies = self.shadow_spectro.spectro.Spectroscopy(D)
        return solution, frequencies



    def loop_expectation_value_q_pauli(self, ms):
        snapshot_expectation_values = (
            self.shadow_spectro.expectation_value_q_pauli((self.clifford[ms], self.Bit_string_array[ms]))
            * self.GAMMA[ms]
        )

        return snapshot_expectation_values
        
        
        
        
    
    def deserialize_circuit(self, qasm_str):
        # VERY UNSTABLE, Lot of gates not recognize->prefere clifford circuit without swap gate
        """Deserialize a QuantumCircuit from JSON."""
        from qiskit.qasm2 import loads
        return loads(qasm_str, custom_instructions=self.custom_instruction_list)


    def send_time_evolve_circuitsV2(self, verbose: bool = True):
        T = np.linspace(0.00, self.Nt * self.dt, self.Nt)
        Gamma = []
        Time_evolve_Clifford_array = []
        job_res=[]
        batch=Batch(backend=self.hardware.backend)
        print(batch.details())
        t0=time.time()
        max_time=batch.details()["max_time"]
        sampler = Sampler(mode=batch)
        for k, t in tqdm(enumerate(T),desc="Generate circuit Time evolution",disable=not verbose,):
            t_batch=time.time()-t0
            print("remaining time before new batch : ",(max_time-600)-t_batch)
            if t_batch>(max_time-600):
                print("Creation of a new_batch")
                batch.close()
                batch=Batch(backend=self.hardware.backend)   
                sampler = Sampler(mode=batch)
                t0=time.time()
                
            circuits_per_t = []
            trotter = TE_PAI(
                self.hamil,
                self.nq,
                self.delta,
                t,
                trotter_steps=self.trotter_steps,
                PAI_error=self.PAI_error,
                N_trotter_max=self.N_trotter_max,
                init_state=self.init_state,
                serialize=True,
                M_sample_max=self.te_pai_shadow_spectro.M_sample_max,
            )

            self.M_sample.append(trotter.M_sample)
            trotter.gen_te_pai_circuits()
            Gamma.append(trotter.GAMMA)
            List_circuit = list(trotter.TE_PAI_Circuits)
            """Main function to execute multiprocessing"""
            
            t0=time.time()
            with mp.Pool(processes=self.num_processes) as pool:
                results = pool.map(self.process_sample, List_circuit)
            print("time to process data ", time.time()-t0)
            
            t0=time.time()
            # Extract results
            snapshot_Clifford_array, snapshot_circuit = zip(*results)
            snapshot_Clifford_array = list(snapshot_Clifford_array)
            for te_pai_circuit in snapshot_circuit:
                for  circ in te_pai_circuit:
                    circuits_per_t.append(circ)
            print("time to extract result data ", time.time()-t0)        
            Time_evolve_Clifford_array.append(snapshot_Clifford_array)

            print("Batch status before running: ", batch.status())
            
            N = len(circuits_per_t) // 4
            t0=time.time()
            list_pubs_for_sampling=[circuits_per_t[:N],circuits_per_t[N:N*2],circuits_per_t[N*2:N*3],circuits_per_t[N*3:]]         
            print("time to generate,  list_pubs_for_sampling", time.time()-t0)        
            
            sub_job=[]
            for n, sub_list in enumerate(list_pubs_for_sampling): 
                job = sampler.run(sub_list, shots=1)  # 1 job per 1 time step
                print("number of pub:", len(sub_list))
                print(f"Done sending sub circuits to backend {n+1}/{4}")
                self.hardware.print_job_info(job)
                if self.hardware.is_fake:  # If the backend is fake, we can get the data directly
                    sub_job.append(self.hardware.get_data_from_results(job.result()))
                else:  # If the backend is QPU, we need to get the job id. We fetch the result from the job id later
                    sub_job.append(job.job_id())
            
            job_res.append(sub_job)
            print(f"Done sending circuits to backend {k+1}/{self.Nt}")
            print("#################################################")
        print("End first Half of the batch, a new batch will be created")
            

        if self.hardware.is_fake:
            # Save jobs to a file
            folder,_=save_to_file(
                self.file_name  +"_res_fake_backend",
                "data/Hardware_simulation/",  # We suppose that the file is run in Frontier_Lab, such as 'python Hardware_simulation/post_process.py'
                use_auto_structure=True,
                format="pickle",
                job=job_res
            )

        else:
            # Write the job ids to a file
            # We suppose that the file is run in Frontier_Lab, such as 'python Hardware_simulation/post_process.py'
            folder,_=save_to_file(
                self.file_name+"_res_QPU",
                "Hardware_simulation/",  # We suppose that the file is run in Frontier_Lab, such as 'python Hardware_simulation/post_process.py'
                use_auto_structure=True,
                format="pickle",
                job=job_res,
            )
            
            
        # Save data
        save_to_file(
            self.file_name+"_data",
            folder,  # We suppose that the file is run in Frontier_Lab, such as 'python Hardware_simulation/post_process.py'
            use_auto_structure=False,
            format="pickle",
            Nt=self.Nt, dt=self.dt, nq=self.nq, Time_evolve_Clifford_array=Time_evolve_Clifford_array, GAMMA=Gamma,M_sample=self.M_sample,
            shadow_size=self.shadow_size)
        return folder
        