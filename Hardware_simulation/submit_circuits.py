# Third-party imports
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


numQs = 25


def J(_):
    return -1 * 0.1


def h(_):
    return -1 * 1


terms = [("ZZ", [k, (k + 1) % numQs], J) for k in range(numQs)]
terms += [("X", [k], h) for k in range(numQs)]


if __name__ == "__main__":
    # # Parameters :
    k: int = 3  # K Pauli observable
    # shadow_size: int = 35  # Shadow size
    shadow_size: int = 3 # Shadow size
    Nt: int = 150  # Number of time steps
    delta: float = np.pi / 2**7 # Trotter Delta
    dt: float = 10/Nt  # Time step

    # Hamiltonian:
    hamil = Hamiltonian(numQs, terms)
    Initial_state = QuantumCircuit(numQs)
    Initial_state.h(i for i in range(numQs))

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
        M_sample_max=400,
        PAI_error=0.09,
        init_state=Initial_state,
    )

    # -------- Hardware settings --------
    # at the first time
    # token = "your API token"
    # is_save = True

    # from the second time
    token = None
    is_save = False

    hardware = Hardware(is_fake=False)
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
