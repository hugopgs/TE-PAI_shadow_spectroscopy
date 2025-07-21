# Written by: Hugo PAGES
# Date: 2024-01-05

# Local application imports
from itertools import combinations, product
import Hamiltonian as Hamil
from TE_PAI_Shadow_Spectroscopy.TE_PAI_Shadow_Spectroscopy import TE_PAI_Shadow_Spectroscopy
from tools_box.plots_functions import *
from tools_box.quantum_tools import *
from tools_box.data_file_functions import *
from Shadow_Spectro import ClassicalShadow, Spectroscopy, ShadowSpectro

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np

# Standard library imports
import warnings
warnings.filterwarnings('ignore')



numQs = 6

def J(t):
    return -2

def J2(t):
    return -0.1
terms = [("ZZ", [k, (k + 1)], J2)for k in range(numQs-1)]
terms += [("X", [k], J) for k in range(numQs)]


if __name__ == "__main__":
    delta = np.pi / (2**5)  # Delta parameter
    shadow_size_TE_PAI = 1
    shadow_size = 1_700
    N_trotter_max = 150
    trotter_step =0.02
    PAI_error=0.001
    k =3
    Nt=60
    dt=3/Nt
    M_sample_max= 1_700
    Noise=[10**(-4),10**(-3)]
    
    hamil = Hamil.Hamiltonian(numQs, terms)
    hamil2 = Hamil.Ising_Hamil(numQs, 0.1, 2)
    
    
    # ground_energy, first_excited_energy_1, ground_components, excited_components_1 =  hamil.get_ground_and_excited_state(30)
    
    # print("energy gap:", np.abs(first_excited_energy_1-ground_energy))

    # Initial_state = ground_components+excited_components_1
    Initial_state = QuantumCircuit(numQs)
    Initial_state.h(i for i in range(numQs-1))
    


    print('dt', dt, 'Nt', Nt,'Tf',dt*Nt, 'N_trotter_max', N_trotter_max, 'shadow_size_TE_PAI', shadow_size_TE_PAI, 'shadow_size', shadow_size, 'PAI_error', PAI_error, 'k', k)

    """TE-PAI Without noise"""
    te_pai_shadow = TE_PAI_Shadow_Spectroscopy(
        hamil, delta, dt, Nt, shadow_size_TE_PAI, K=k, trotter_steps=trotter_step, N_trotter_max=N_trotter_max, 
        PAI_error=PAI_error, init_state=Initial_state, M_sample_max=M_sample_max)

    # TE_PAI Shadow spectro
    frequencies_TE_PAI, solution_TE_PAI =te_pai_shadow.main_te_pai_shadow_spectro(
        density_matrix=True, serialize=True)
    
    file_name = f"Heisenberg_Hamil_nq{numQs}_J{J(1)}_Nt{Nt}_dt{dt:.2}_NTrot{N_trotter_max}_NsTE{shadow_size_TE_PAI}_res_TE_PAI_no_noise"
    # save data
    folder, _ = save_to_file(file_name, "data/TROTTER_VS_TE_PAI", format="pickle",
                             Nt=Nt, dt=dt, solution_TE_PAI=solution_TE_PAI, frequencies_TE_PAI=frequencies_TE_PAI,Energy_gap=hamil.energy_gap(), J=J(1))
    plot_spectre(frequencies_TE_PAI, solution_TE_PAI, save_as="TE_PAI_spectrum", Folder=folder)

    avg_depth=np.mean(te_pai_shadow.depth)
    
    N_trotter_steps_same_depth= 16
    print("N_trotter_steps from depth: ", N_trotter_steps_same_depth)
    
    
    
    """TE-PAI With noise"""

    te_pai_shadow = TE_PAI_Shadow_Spectroscopy(
        hamil, delta, dt, Nt, shadow_size_TE_PAI, K=k, trotter_steps=trotter_step, N_trotter_max=N_trotter_max, 
        PAI_error=PAI_error, init_state=Initial_state, M_sample_max=M_sample_max, noise_coef=Noise)

    
        # TE_PAI Shadow spectro
    frequencies_TE_PAI_noise, solution_TE_PAI_noise = te_pai_shadow.main_te_pai_shadow_spectro(
        density_matrix=False, serialize=True)
    
    file_name = f"Heisenberg_Hamil_nq{numQs}_J{J(1)}_Nt{Nt}_dt{dt:.2}_NTrot{N_trotter_max}_NsTE{shadow_size_TE_PAI}_noise{Noise}_res_TE_PAI_noisy"
    # save data
    save_to_file(file_name, folder_name=folder, format="pickle", use_auto_structure=False,
                             Nt=Nt, dt=dt, solution_TE_PAI_noise=solution_TE_PAI_noise, frequencies_TE_PAI_noise=frequencies_TE_PAI_noise,Energy_gap=hamil.energy_gap(), J=J(1))
   
    path_noise="/root/work/data/TROTTER_VS_TE_PAI/2025_07_08/Acquisition_1/Heisenberg_Hamil_nq10_J-1_Nt60_dt0.02_NTrot25_NsTE1_noise[0.0001, 0.001]_res_TE_PAI_noisy.pickle"
    path_no_noise="/root/work/data/TROTTER_VS_TE_PAI/2025_07_08/Acquisition_1/Heisenberg_Hamil_nq10_J-1_Nt60_dt0.02_NTrot25_NsTE1_res_TE_PAI_no_noise.pickle"
    Nt, dt, solution_TE_PAI_noise, frequencies_TE_PAI_noise,_, _=get_data_file(path_noise)
    Nt, dt, solution_TE_PAI, frequencies_TE_PAI,_, _=get_data_file(path_no_noise)
    
    folder=get_file_parent_folder_path(path_noise)
    """TROTTER """
    shadow = ClassicalShadow()
    spectro = Spectroscopy(Nt, dt)
    shadow_spectro = ShadowSpectro(shadow, spectro, numQs, k, shadow_size)

    # Shadow spectro
    frequencies_Trotter, solution_Trotter = shadow_spectro.shadow_spectro(
        hamil, init_state=Initial_state, N_Trotter_steps=N_trotter_max, density_matrix=False, serialize=True, multiprocessing=True)

    # avg_depth=np.mean(te_pai_shadow.depth)
    
    # N_trotter_steps_same_depth= hamil2.get_trotter_steps_from_depth(avg_depth)
    # print("N_trotter_steps : ", N_trotter_steps_same_depth)
    path_noise="/root/work/data/TROTTER_VS_TE_PAI/2025_07_08/Acquisition_1/Heisenberg_Hamil_nq10_J-1_Nt60_dt0.02_NTrot25_NsTE1_noise[0.0001, 0.001]_res_TE_PAI_noisy.pickle"
    path_no_noise="/root/work/data/spectrum_nq4_J1_Nt90_dt0.11_NTrot700_Msample500_NsTE1_Ns50_delta0.025.pickle"
    Nt, dt, solution_TE_PAI_noise, frequencies_TE_PAI_noise,_, _=get_data_file(path_noise)
    Nt, dt, solution_TE_PAI, frequencies_TE_PAI,_, _=get_data_file(path_no_noise)
    
    folder=get_file_parent_folder_path(path_noise)
    
    N_trotter_steps_same_depth=15
    
    frequencies_Trotter_same_depth, solution_Trotter_same_depth = shadow_spectro.shadow_spectro(
        hamil, init_state=Initial_state, N_Trotter_steps=N_trotter_steps_same_depth, density_matrix=False, serialize=True, multiprocessing=True)

    
    shadow = ClassicalShadow(noise_error=Noise)
    spectro = Spectroscopy(Nt, dt)
    shadow_spectro = ShadowSpectro(shadow, spectro, numQs, k, shadow_size)
    # Shadow spectro
    frequencies_Trotter_noise, solution_Trotter_noise = shadow_spectro.shadow_spectro(
        hamil, init_state=Initial_state, N_Trotter_steps=N_trotter_max, density_matrix=False, serialize=True, multiprocessing=True)

    frequencies_Trotter_same_depth_noise, solution_Trotter_same_depth_noise = shadow_spectro.shadow_spectro(
        hamil, init_state=Initial_state, N_Trotter_steps=N_trotter_steps_same_depth, density_matrix=False, serialize=True, multiprocessing=True)

    """SAVE AND PLOT"""
   
    file_name = f"Heisenberg_Hamil_nq{numQs}_J{J(1)}_Nt{Nt}_dt{dt:.2}_NTrot{N_trotter_max}_NsTE{shadow_size_TE_PAI}_res"
    # save data
    

    
    
    save_to_file(file_name, folder_name=folder, format="pickle",use_auto_structure=False,
                             Nt=Nt, dt=dt, solution_TE_PAI_noise=solution_TE_PAI_noise, 
                             frequencies_TE_PAI_noise=frequencies_TE_PAI_noise,
                             solution_TE_PAI=solution_TE_PAI, frequencies_TE_PAI=frequencies_TE_PAI,
                             solution_Trotter_noise=solution_Trotter_noise, frequencies_Trotter_noise=frequencies_Trotter_noise,
                             solution_Trotter=solution_Trotter, frequencies_Trotter=frequencies_Trotter,
                             solution_Trotter_same_depth=solution_Trotter_same_depth, frequencies_Trotter_same_depth=frequencies_Trotter_same_depth,
                             solution_Trotter_same_depth_noise=solution_Trotter_same_depth_noise,
                             frequencies_Trotter_same_depth_noise=frequencies_Trotter_same_depth_noise,
                             Energy_gap=hamil.energy_gap())

   
    plot_multiple_data([frequencies_TE_PAI, frequencies_TE_PAI_noise, frequencies_Trotter, frequencies_Trotter_noise,frequencies_Trotter_same_depth,
                        frequencies_Trotter_same_depth_noise],
                       [solution_TE_PAI, solution_TE_PAI_noise, solution_Trotter, solution_Trotter_noise, solution_Trotter_same_depth,
                        solution_Trotter_same_depth_noise],
                       color=["red","red","blue","blue", "green","green"],
                       linestyle=['-','--'],
                       labels=["TE_PAI without noise", "TE_PAI with noise", "Trotter without noise", "Trotter with noise", "Trotter_same_depth", "Trotter_same_depth_noise"],
                       title="TE_PAI Shadow spectroscopy with/without noise \n of a Transverse Hamiltonian",
                       save_as=file_name+"spectre_Multi", Folder=folder)