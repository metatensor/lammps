import ase.build
import ase.units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import torch
from metatomic.torch.ase_calculator import MetatomicCalculator

from flashmd import get_pretrained
from flashmd.ase.langevin import Langevin


# Choose your time step (go for 10-30x what you would use in normal MD for your system)
time_step = 8  # 64 fs; also available: 1, 2, 4, 8, 16, 32, 128 fs

# Create a structure and initialize velocities
import ase.io
atoms = ase.io.read("water-128.xyz")
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
atoms.set_velocities(  # it is generally a good idea to remove any net velocity
    atoms.get_velocities() - atoms.get_momenta().sum(axis=0) / atoms.get_masses().sum()
)

# Load models
device="cuda" if torch.cuda.is_available() else "cpu"
energy_model, flashmd_model = get_pretrained("pet-omatpes", time_step)  

calculator = MetatomicCalculator(energy_model, device=device)
atoms.calc = calculator

# Run MD
dyn = Langevin(
    atoms=atoms,
    timestep=time_step*ase.units.fs,
    temperature_K=300,
    time_constant=100*ase.units.fs,
    model=flashmd_model,
    device=device
)
dyn.attach(lambda: print(atoms.get_temperature()), interval=1)
dyn.run(1000)
