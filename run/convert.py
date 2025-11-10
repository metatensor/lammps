import ase.io


atoms = ase.io.read("water-128.xyz")
ase.io.write("water-128.lmp", atoms, format="lammps-data")
