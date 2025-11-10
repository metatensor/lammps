import ase.build


molecule = ase.build.molecule("H2O")
molecule.cell = [100.0, 100.0, 100.0]
molecule.pbc = [True, True, True]
molecule.center()
ase.io.write("water-1.xyz", molecule)
ase.io.write("water-1.lmp", molecule, format="lammps-data")
