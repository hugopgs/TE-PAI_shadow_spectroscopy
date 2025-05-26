# Third-party imports
from itertools import product
import numpy as np

# Local application imports
from TE_PAI_Shadow_Spectro_Hardware import TE_PAI_Shadow_Spectro_Hardware
from Hardware import Hardware
from TE_PAI_Shadow_Spectroscopy import TE_PAI_Shadow_Spectroscopy
from Shadow_Spectro.ShadowSpectro import ShadowSpectro
from Shadow_Spectro.Spectroscopy import Spectroscopy
from Shadow_Spectro.ClassicalShadow import ClassicalShadow
from Hamiltonian.Hamiltonian import Hamiltonian
from tools_box.quantum_tools import *


numQs = 10

def J(t):
    return 1


terms = [
    (gate, [k, k + 1], J if gate ==
        "XX" else J if gate == "YY" else J)
    for k, gate in product(range(numQs-1), ["XX", "YY", "ZZ"])
    ]



if __name__ == "__main__":
    # # Parameters :
    k: int = 3  # K Pauli observable
    # shadow_size: int = 35  # Shadow size
    shadow_size: int =  5 # Shadow size
    Nt: int = 130 # Number of time steps
    delta: float = np.pi / 2**7 # Trotter Delta
    dt: float = 11/Nt  # Time step

    # Hamiltonian:
    hamil = Hamiltonian(numQs, terms)
    ground_energy, first_excited_energy, ground_components, excited_components =  hamil.get_ground_and_excited_state()
    Initial_state = ground_components + excited_components

    # Creation Shadow spectro
    shadow = ClassicalShadow()
    spectro = Spectroscopy(Nt, dt)
    shadow_spectro = ShadowSpectro(shadow, spectro, numQs, k, shadow_size)
    te_pai_shadow = TE_PAI_Shadow_Spectroscopy(
        hamil,
        delta,
        dt,
        Nt,
        shadow_size,
        K=k,
        trotter_steps=0.012,
        N_trotter_max=900,
        M_sample_max=500,
        PAI_error=0.1,
        init_state=Initial_state,
    )

    # -------- Hardware settings --------
    # at the first time
    # token = "your API token"
    # is_save = True

    # from the second time
    # token = "your API token"
    token = None
    is_save = False

    hardware = Hardware(token, is_fake=False)
    # is_fake=False: QPU
    # -------- Hardware settings --------

    # Shadow_spectro Hardware
    # shadow_hardware = ShadowSpectro_Hardware(hardware, shadow_spectro)

    # TE_PAI_Shadow_spectro Hardware
    TE_PAI_Shadow_Hardware = TE_PAI_Shadow_Spectro_Hardware(
        hardware, te_pai_shadow)

    # """SCRIPT FOR TE_PAI SHADOW SPECTROSCOPY"""
    # bit_string_array, Time_evolve_Clifford_array, GAMMA = (
    #     TE_PAI_Shadow_Hardware.send_time_evolve_circuits(verbose=True)
    # )
    hardware
    TE_PAI_Shadow_Hardware.send_time_evolve_circuits(verbose=True)
