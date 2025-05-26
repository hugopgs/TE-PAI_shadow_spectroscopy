# Written by: Hugo PAGES
# Date: 2024-01-05

# Local application imports
from itertools import product
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





################# Hamiltonian definition ################# 
numQs = 10
def J(t):
    return 1

terms = [
    (gate, [k, k + 1], J if gate ==
        "XX" else J if gate == "YY" else J)
    for k, gate in product(range(numQs-1), ["XX", "YY", "ZZ"])  #Heisenberg Hamiltonian
    ]
################# Hamiltonian definition ################# 


if __name__ == "__main__":
    
    ################# Parameters ################# 
    delta : float = (np.pi / (2**7)) 
    shadow_size_TE_PAI : int = 5
    shadow_size : int = 50
    N_trotter_max : int = 900
    trotter_step : float = 0.002
    PAI_error : float = 0.1
    k : int = 3
    Nt : int = 150
    dt : float = 10/Nt
    M_sample_max : int =500
    folder : str ="data"
    ################# Parameters ################# 
    
    
    ######## Generation Initial State ########
    hamil = Hamil.Hamiltonian(numQs, terms)
    ground_energy, first_excited_energy, ground_components, excited_components=hamil.get_ground_first_excited_state()
    Initial_state=sum_state_vectors(excited_components, ground_components, numQs)
    ######## Generation Initial State ########
    
    ######## Instance TE PAI Shadow spectroscopy #######
    te_pai_shadow = TE_PAI_Shadow_Spectroscopy(
        hamil, delta, dt, Nt, shadow_size_TE_PAI, K=k, trotter_steps=trotter_step, N_trotter_max=N_trotter_max, PAI_error=PAI_error, init_state=Initial_state, M_sample_max=M_sample_max)
    shadow = ClassicalShadow()
    spectro = Spectroscopy(Nt, dt)
    shadow_spectro = ShadowSpectro(shadow, spectro, numQs, k, shadow_size)
     ######## Instance TE PAI Shadow spectroscopy #######
    


    ################  Simulations ################ 
    solution_TE_PAI, frequencies_TE_PAI = te_pai_shadow.main_te_pai_shadow_spectro(
        density_matrix=False, serialize=True)  #TE_PAI shadow spectro 
    solution, frequencies = shadow_spectro.shadow_spectro(
        hamil, init_state=Initial_state, N_Trotter_steps=1400, density_matrix=False, serialize=True, multiprocessing=True)  #shadow spectro
    ################  Simulations ################ 




    ############# Save Data ####################
    file_name = f"Heisenberg_Hamil_nq{numQs}_J{J(1)}_Nt{Nt}_dt{dt:.2}_NTrot{N_trotter_max}_NsTE{shadow_size_TE_PAI}_Ns{shadow_size}_PAIerror{PAI_error}_delta{delta:.2}_M_sample_max{M_sample_max}"
    # save data
    save_to_file(file_name, folder_name=folder, format="pickle",use_auto_structure=False,
                             Nt=Nt, dt=dt, solution=solution, frequencies=frequencies,
                             solution_TE_PAI=solution_TE_PAI, frequencies_TE_PAI=frequencies_TE_PAI,
                             Energy_gap=hamil.energy_gap(), J=J(1))

    # plot and save
    plot_spectre(frequencies_TE_PAI, solution_TE_PAI, label="TE_PAI shadow",
                 save_as=file_name+"_spectre_TE_PAI", Folder=folder)
    plot_spectre(frequencies, solution, save_as=file_name +
                 "spectre_shadow", Folder=folder)
    plot_multiple_data([frequencies_TE_PAI, frequencies], [solution_TE_PAI, solution], labels=["TE_PAI", "Shadow"],
                       title="TE_PAI Shadow spectroscopy and Shadow spectroscopy \n of a Heisenberg Hamiltonian",
                       save_as=file_name+"spectre_Multi", Folder=folder)

    # plt.show()
