# Written by: Hugo PAGES 
# Date: 2024-01-05

# Standard library imports
import sys
import os

folder = os.path.dirname(os.path.abspath(os.path.dirname(sys.argv[0])))
sys.path.append(folder)

#Third-party imports
import matplotlib.pyplot as plt
import numpy as np

#Local application imports
from tools_box.plots_functions import *
from tools_box.quantum_tools import *
from tools_box.data_file_functions import get_data_file, get_file_parent_folder_path

if __name__ == "__main__":
    numQs : int = 10
    path : str =""
    Folder : str = get_file_parent_folder_path(path)
    
    Nt, dt, solution_Shadow,frequencies_shadow,solution_TE_PAI, frequencies_TE_PAI,Energy_gap, J=get_data_file(path) #get data from file
    
    
    pos = np.argsort(np.abs(solution_TE_PAI))[::-1]
    Energy_gap_exp=frequencies_TE_PAI[pos[0]]
    print("Energy gap exp:", Energy_gap_exp)
    closest_energyGap=closest_value(Energy_gap,  Energy_gap_exp)
    print("closest energy gap", closest_energyGap)
    
    freq_resample, solution_resample=resample_points(frequencies_TE_PAI,solution_TE_PAI,100) #linear interpolation
    freq_resample_shadow, solution_resample_shadow=resample_points(frequencies_shadow,solution_Shadow,100) #linear interpolation

    #filtrage
    filtre= chi_asymmetric(freq_resample,frequencies_TE_PAI[pos[0]],sigma_L=4,sigma_R=25 )  
    # filtre3_= chi_asymmetric(freq_resample_shadow,frequencies_TE_PAI[pos[0]],sigma_L=4,sigma_R=25 )  
    plot_spectre(freq_resample,solution_resample*filtre, Energy_gap=[closest_energyGap],save_as="filtered_data_TE_PAI", Folder=Folder)
    
    
    #plot
    plot_multiple_data([freq_resample, freq_resample_shadow], [(solution_resample), (solution_resample_shadow)],
                       labels=["TE_PAI", "Shadow"], title="TE_PAI Shadow spectroscopy and Shadow spectroscopy \n of a Transverse Ising Hamiltonian ", Energy_gap=[closest_energyGap],
                       save_as="filtered_data_multiple_vline", Folder=Folder
                       )
    
    plot_multiple_data([freq_resample, freq_resample_shadow], [(solution_resample*filtre), (solution_resample_shadow*filtre)],
                       labels=["TE_PAI", "Shadow"], title="TE_PAI Shadow spectroscopy and Shadow spectroscopy \n of a Transverse Ising Hamiltonian ",
                       save_as="filtered_data_multiple", Folder=Folder
                       )
    plt.show()