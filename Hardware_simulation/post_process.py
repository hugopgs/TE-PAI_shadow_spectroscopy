# Written by: Yuto Morohoshi, udpdated by Hugo PAGES

# Standard library imports
import multiprocessing as mp
import gc
import collections

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from TE_PAI_Shadow_Spectro_Hardware import TE_PAI_Shadow_Spectro_Hardware
from Hardware import Hardware
from TE_PAI_Shadow_Spectroscopy import TE_PAI_Shadow_Spectroscopy
from Shadow_Spectro.ShadowSpectro import ShadowSpectro
from Shadow_Spectro.Spectroscopy import Spectroscopy
from Shadow_Spectro.ClassicalShadow import ClassicalShadow
from Hamiltonian.Hamiltonian import Hamiltonian
from tools_box.quantum_tools import *
from tools_box.data_file_functions import (
    get_data_file,
    get_keywords_pickle_file,
    save_to_file,get_file_parent_folder_path
)
from tools_box.plots_functions import plot_spectre, plot_multiple_data

numQs = 4


def J(_):
    return -1 * 0.8


def h(_):
    return -1 * 0.5


terms = [("ZZ", [k, (k + 1) % numQs], J) for k in range(numQs)]
terms += [("X", [k], h) for k in range(numQs)]

if __name__ == "__main__":
    
    # Retrieve data from the Hardware acquisition :
    TE_PAI_data="/root/work/data/2025_05_07/Acquisition_1/Heisenberg_Hamil_nq4_J-0.1_Nt150_dt0.067_NTrot400_NsTE1_Ns50.pickle"
    path_data="/root/work/data/2025_05_08/2025_05_08/Acquisition_1/nq4_Nt150_dt0.067_NTrot400_NsTE5_data.pickle"
    path_res_jobs="/root/work/data/2025_05_08/2025_05_08/Acquisition_1/nq4_Nt150_dt0.067_NTrot400_NsTE5_res_QPU.pickle"
    Folder_path=get_file_parent_folder_path(path_data)
    Nt, dt, nq, Time_evolve_Clifford_array, GAMMA,M_sample, shadow_size=get_data_file(path_data)
    
    # Parameters :
    k: int = 3  # K Pauli observable
    delta: float = np.pi / 2**7  # Trotter Delta
    file_name = f"nq{nq}_Nt{Nt}_dt{dt:.2}_NsTE{shadow_size}"
    hamil=Hamiltonian(nq, terms)

    # Creation Shadow spectro
    shadow = ClassicalShadow()
    spectro = Spectroscopy(Nt, dt)
    shadow_spectro = ShadowSpectro(shadow, spectro, nq, k, shadow_size)
    te_pai_shadow = TE_PAI_Shadow_Spectroscopy(hamil,delta,dt,Nt,shadow_size,K=k)

    job_id= get_data_file(path_res_jobs)

    # -------- Hardware settings --------
    token = ""  # Token for the hardware
    is_save = True
    hardware = Hardware(
        token=token)  # Hardware class to retrieve data from the job.id
   
    # -------- Hardware settings --------


    TE_PAI_Shadow_Hardware = TE_PAI_Shadow_Spectro_Hardware(hardware, te_pai_shadow) 

    # res_fake_backend=[]
    # for sub_list in res_bit_array: 
    #     for sub_sub_list in sub_list : 
    #         for bit in sub_sub_list:
    #             res_fake_backend.append(bit)
    # print(res_fake_backend)
    # bit_array = [bit for sublist in res_bit_array for bit in sublist]
    # print(f"Number of bit strings after flattening: {len(res_fake_backend)}")  
    
        
    TE_PAI_Shadow_Hardware.M_sample = M_sample #set the M_sample for post processing 
    
    
    # -------- Post processing --------
    
    solution, frequency = TE_PAI_Shadow_Hardware.get_data_matrix_from_hardware(
        job_id, GAMMA, Time_evolve_Clifford_array,is_density_matrix=False, is_verbose=True
    )
    # -------- Post processing --------
    
    
    
    
    # -------- save and plot --------
    save_to_file(name=file_name+"res_spectro", folder_name=Folder_path, use_auto_structure=False,solution=solution, frequency=frequency )
    plot_spectre(frequency, solution, save_as=file_name+"_Spetrum_Hardware", Folder=Folder_path)
    plt.show()
     # -------- save and plot --------
     
     
    
    Nt, dt, solution_Shadow,frequencies_shadow,solution_TE_PAI, frequencies_TE_PAI,Energy_gap, J=get_data_file(TE_PAI_data)
    plot_multiple_data([frequency,frequencies_shadow, frequencies_TE_PAI],[solution,solution_Shadow,solution_TE_PAI], labels=["QPU TE_PAI shadow", "CPU Shadow","CPU TE_PAI shadow"],
                       save_as="Comparison_dimulation", Folder=Folder_path)
    # plt.show()
    
    #------ Interpolation + Filtering --------
    pos = np.argsort(np.abs(solution))[::-1]
    freq_resample, solution_resample=resample_points(frequency,solution,100) #linear interpolation
    freq_resample_TE_PAI, solution_resample_TE_PAI=resample_points(frequencies_TE_PAI,solution_TE_PAI,100)
    filtre= chi_asymmetric(freq_resample,frequency[pos[0]],sigma_L=2,sigma_R=12 )  
    
    solution_resample=solution_resample*filtre
    solution_resample_TE_PAI=solution_resample_TE_PAI*filtre
    
    
    plot_multiple_data([freq_resample,frequencies_shadow, freq_resample_TE_PAI],[solution_resample,solution_Shadow,solution_resample_TE_PAI], labels=["QPU TE_PAI shadow", "CPU Shadow","CPU TE_PAI shadow"],
                       save_as="Comparison_simulation", Folder=Folder_path)
    plt.show()
    