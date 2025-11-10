import ase.build
import ase.units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import torch
from metatomic.torch.ase_calculator import MetatomicCalculator

from flashmd import get_pretrained
from flashmd.ase.velocity_verlet import VelocityVerlet


# Choose your time step (go for 10-30x what you would use in normal MD for your system)
time_step = 8  # 64 fs; also available: 1, 2, 4, 8, 16, 32, 128 fs

# Create a structure and initialize velocities
atoms = ase.build.molecule("H2O")
import ase.io
atoms.cell = [100.0, 100.0, 100.0]
atoms.pbc = [True, True, True]
atoms.center()
ase.io.write("water-1.xyz", atoms)
ase.io.write("water-1.lmp", atoms, format="lammps-data")
# MaxwellBoltzmannDistribution(atoms, temperature_K=300)
atoms.set_velocities(  # it is generally a good idea to remove any net velocity
    atoms.get_velocities() - atoms.get_momenta().sum(axis=0) / atoms.get_masses().sum()
)

# Load models
device="cuda" if torch.cuda.is_available() else "cpu"
energy_model, flashmd_model = get_pretrained("pet-omatpes", time_step)  

calculator = MetatomicCalculator(energy_model, device=device)
atoms.calc = calculator

# Run MD
dyn = VelocityVerlet(
    atoms=atoms,
    timestep=time_step*ase.units.fs,
    model=flashmd_model,
    rescale_energy=False,
    random_rotation=False,
)

print(atoms.get_momenta())
dyn.run(1)
print(atoms.get_momenta())
