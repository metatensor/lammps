from metatomic.torch import load_atomistic_model, ModelOutput, ModelEvaluationOptions
from metatomic.torch import systems_to_torch
import ase.io
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
from metatensor.torch import TensorMap, TensorBlock, Labels
import torch
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


model = load_atomistic_model("flashmd.pt").to("cuda")
atoms = ase.io.read("water-1.xyz")
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
system = systems_to_torch(atoms, device="cuda")
system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

system.add_data(
    "masses",
    TensorMap(
        Labels.single(),
        [
            TensorBlock(
                values=torch.tensor(atoms.get_masses()[:, None], dtype=torch.float32),
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor([[0, i] for i in range(len(system))]),
                ),
                components=[],
                properties=Labels.single(),
            )
        ]
    ).to(system.device)
)

system.add_data(
    "momenta",
    TensorMap(
        Labels.single(),
        [
            TensorBlock(
                values=torch.tensor(atoms.get_momenta()[:, :, None], dtype=torch.float32),
                samples=Labels(
                    ["system", "atom"],
                    torch.tensor([[0, i] for i in range(len(system))]),
                ),
                components=[Labels(["xyz"], torch.tensor([[0], [1], [2]]))],
                properties=Labels.single(),
            )
        ]
    ).to(system.device)
)

# print(system.positions)
# print(system.get_data("masses").block().values)
print(system.get_data("momenta").block().values)
# print(system.types)
# print(system.cell)
# print(system.pbc)

options = ModelEvaluationOptions(
    length_unit="angstrom",
    outputs={
        "positions": ModelOutput(unit="angstrom", per_atom=True),
        "momenta": ModelOutput(unit="(eV*u)^1/2", per_atom=True),
    }
)

outputs = model([system], options, check_consistency=True)
# print(outputs["positions"].block().values)
print(outputs["momenta"].block().values)
