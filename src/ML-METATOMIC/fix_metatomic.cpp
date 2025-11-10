// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
#include "pair_metatomic.h"
#include "metatomic_types.h"
#include "metatomic_system.h"

#include "fix_metatomic.h"

#include "atom.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "update.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "comm.h"

#include<vector>
#include <iostream>

#include <metatomic/torch.hpp>
#include <metatensor/torch.hpp>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixMetatomic::FixMetatomic(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  std::string energy_unit;
  std::string length_unit;
  if (strcmp(update->unit_style, "real") == 0) {
      length_unit = "angstrom";
      energy_unit = "kcal/mol";
  } else if (strcmp(update->unit_style, "metal") == 0) {
      length_unit = "angstrom";
      energy_unit = "eV";
  } else if (strcmp(update->unit_style, "si") == 0) {
      length_unit = "meter";
      energy_unit = "joule";
  } else if (strcmp(update->unit_style, "electron") == 0) {
      length_unit = "Bohr";
      energy_unit = "Hartree";
  } else {
      error->all(FLERR, "unsupported units '{}' for fix metatomic ", update->unit_style);
  }

  if (narg < 4) error->all(FLERR, "Illegal fix metatomic command");

  bool types_are_set = false;
  model_path = arg[3];
  std::string energy_model_path;
  bool rescale_energy = false;
  std::vector<int> parsed_types;

  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "types") == 0) {
      types_are_set = true;
      // try to run std::stoi on all the following arguments; if an exception is thrown,
      // we stop parsing the types
      int current_num_types = 0;
      iarg++;
      while (iarg < narg) {
        int type = -1;
        try {
          type = std::stoi(arg[iarg]);
          iarg++;
        } catch (const std::invalid_argument &) {
          break;  // stop parsing types on invalid argument to std::stoi
        }
        if (type <= 0) {
          error->all(FLERR, "Illegal fix metatomic command: type {} should be > 0", type);
        }
        parsed_types.push_back(type);
        current_num_types++;
        if (current_num_types > atom->ntypes) {
          error->all(FLERR, "Illegal fix metatomic command: too many types specified");
        }
      }
    } else if (strcmp(arg[iarg], "energy") == 0) {
      if (iarg + 1 > narg) error->all(FLERR, "Illegal fix metatomic command");
      energy_model_path = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "device") == 0) {
      if (iarg + 1 > narg) error->all(FLERR, "Illegal fix metatomic command");
      requested_device = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "rescale_energy") == 0) {
      if (iarg + 1 > narg) error->all(FLERR, "Illegal fix metatomic command");
      if (strcmp(arg[iarg + 1], "on") == 0) {
        rescale_energy = true;
      } else if (strcmp(arg[iarg + 1], "off") == 0) {
        rescale_energy = false;
      } else {
        error->all(FLERR, "Illegal fix metatomic command: expected 'on' or 'off' after 'rescale_energy'");
      }
      iarg += 2;
    } else {
      error->all(FLERR, "Illegal fix metatomic command");
    }
  }
  if (!types_are_set) {
    error->all(FLERR, "Illegal fix metatomic command: no types specified");
  }
  if ((int)parsed_types.size() != atom->ntypes) {
    error->all(FLERR, "Illegal fix metatomic command: number of types does not match number of atom types");
  }

  // Allocate and fill the type-mapping (1-based indexing)
  type_mapping = memory->create(type_mapping, atom->ntypes + 1, "FixMetatomic:type_mapping");
  for (int i = 1; i <= atom->ntypes; i++) {
    type_mapping[i] = parsed_types[i - 1];
  }

  this->mta_data = new PairMetatomicData(std::move(length_unit), std::move(energy_unit), true);

  time_integrate = 1;  // this tells LAMMPS that this fix advances simulation time
  // Note: for now we don't allow dynamic groups (dynamic_group_allow variable)
}

FixMetatomic::~FixMetatomic() {
  memory->destroy(type_mapping);
}

/* ---------------------------------------------------------------------- */

int FixMetatomic::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= POST_FORCE;
  mask |= FINAL_INTEGRATE;
  // mask |= INITIAL_INTEGRATE_RESPA;  // ??
  // mask |= FINAL_INTEGRATE_RESPA;  // ??
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixMetatomic::init()
{
  if (!type_mapping) {
      error->all(FLERR, "FixMetatomic internal error: type_mapping not initialized");
  }

  const char *extensions_directory = nullptr;

  dt = update->dt;
  // TODO: what to do with units if not metal????
  mta_data->load_model(this->lmp, model_path.c_str(), extensions_directory);

  // Select the device to use based on the model's preference, the user choice
  // and what's available.
  this->pick_device(&mta_data->device, requested_device.c_str());

  // move all data to the correct device
  mta_data->model->to(mta_data->device);
  mta_data->selected_atoms_values = mta_data->selected_atoms_values.to(mta_data->device);

  auto message = "Running simulation on " + mta_data->device.str() + " device with " + mta_data->capabilities->dtype() + " data";
  if (screen) {
      fprintf(screen, "%s\n", message.c_str());
  }
  if (logfile) {
      fprintf(logfile,"%s\n", message.c_str());
  }

  // get the model's interaction range
  auto range = mta_data->capabilities->engine_interaction_range(mta_data->evaluation_options->length_unit());
  if (range < 0) {
      error->all(FLERR, "interaction_range is negative for this model");
  } else if (!std::isfinite(range)) {
      if (comm->nprocs > 1) {
          error->all(FLERR,
              "interaction_range is infinite for this model, "
              "using multiple MPI domains is not supported"
          );
      }

      // determine the maximal cutoff in the NL
      auto requested_nl = mta_data->model->run_method("requested_neighbor_lists");
      for (const auto& ivalue: requested_nl.toList()) {
          auto options = ivalue.get().toCustomClass<metatomic_torch::NeighborListOptionsHolder>();
          auto cutoff = options->engine_cutoff(mta_data->evaluation_options->length_unit());

          mta_data->max_cutoff = std::max(mta_data->max_cutoff, cutoff);
      }
  } else {
      mta_data->max_cutoff = range;
  }

  // Initialize metatensor system object
  auto options = MetatomicSystemOptions{
    this->type_mapping,
    mta_data->max_cutoff,
    mta_data->check_consistency,
    !(mta_data->non_conservative),
  };
  this->system_adaptor = std::make_unique<MetatomicSystemAdaptor>(lmp, options);

  // We ask LAMMPS for a full neighbor lists because we need to know about
  // ALL pairs, even if options->full_list() is false. We will then filter
  // the pairs to only include each pair once where needed.
  auto request = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
  request->set_cutoff(mta_data->max_cutoff);

  // Translate from the metatomic neighbor lists requests to LAMMPS neighbor
  // lists requests.
  auto requested_nl = mta_data->model->run_method("requested_neighbor_lists");
  for (const auto& ivalue: requested_nl.toList()) {
      auto options = ivalue.get().toCustomClass<metatomic_torch::NeighborListOptionsHolder>();
      auto cutoff = options->engine_cutoff(mta_data->evaluation_options->length_unit());
      assert(cutoff <= mta_data->max_cutoff);

      this->system_adaptor->add_nl_request(cutoff, options);
  }
}

std::vector<torch::DeviceType> FixMetatomic::available_devices() {
    auto devices = std::vector<torch::DeviceType>();
    for (const auto& supported: this->mta_data->capabilities->supported_devices) {
        if (supported == "cpu") {
            devices.push_back(torch::kCPU);
        } else if (supported == "cuda" && torch::cuda::is_available()) {
            devices.push_back(torch::kCUDA);
        } else if (supported == "mps") {
            #if TORCH_VERSION_MAJOR >= 2
            if (torch::mps::is_available()) {
                devices.push_back(torch::kMPS);
            }
            #endif
        } else {
            error->warning(FLERR,
                "the model declared support for unknown device '{}', it will be ignored", supported
            );
        }
    }

    if (devices.empty()) {
        error->all(FLERR,
            "failed to find a valid device for this model: "
            "the model supports {}, none of these where available",
            torch::str(this->mta_data->capabilities->supported_devices)
        );
    }

    return devices;
}

void FixMetatomic::pick_device(torch::Device* device, const char* requested) {
    auto available_devices = this->available_devices();

    auto picked_device_type = torch::kCPU;
    if (requested == nullptr) {
        // no user request, pick the device the model prefers
        picked_device_type = available_devices[0];
    } else {
        bool found_requested_device = false;
        for (const auto& device_type: available_devices) {
            if (device_type == torch::kCPU && strcmp(requested, "cpu") == 0) {
                picked_device_type = device_type;
                found_requested_device = true;
                break;
            } else if (device_type == torch::kCUDA && strcmp(requested, "cuda") == 0) {
                picked_device_type = device_type;
                found_requested_device = true;
                break;
            } else if (device_type == torch::kMPS && strcmp(requested, "mps") == 0) {
                picked_device_type = device_type;
                found_requested_device = true;
                break;
            }
        }

        if (!found_requested_device) {
            error->all(FLERR,
                "failed to find requested device ({}): it is either "
                "not supported by this model or not available on this machine",
                requested
            );
        }
    }

    if (picked_device_type == torch::kCUDA) {
        // distribute GPUs between multiple MPI processes on the same node

        // (1) get a MPI communicator for all processes on the current node
        MPI_Comm local;
        MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
        // (2) get the rank of this MPI process on the current node
        int local_rank;
        MPI_Comm_rank(local, &local_rank);

        int size;
        MPI_Comm_size(local, &size);
        if (size < torch::cuda::device_count()) {
            if (comm->me == 0) {
                error->warning(FLERR,
                    "found {} CUDA-capable GPUs, but only {} MPI processes on the current node; the remaining GPUs will not be used",
                    torch::cuda::device_count(), size
                );
            }
        }

        // (3) split GPUs between node-local processes using round-robin allocation
        int gpu_to_use = local_rank % torch::cuda::device_count();
        *device = torch::Device(picked_device_type, gpu_to_use);
    } else {
        *device = torch::Device(picked_device_type);
    }
}

void FixMetatomic::init_list(int id, NeighList *ptr) {
  mta_list = ptr;
}

void FixMetatomic::initial_integrate(int /*vflag*/)
{
  // update v and x of atoms in group

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;

  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  auto dtype = torch::kFloat64;
  if (mta_data->capabilities->dtype() == "float64") {
      dtype = torch::kFloat64;
  } else if (mta_data->capabilities->dtype() == "float32") {
      dtype = torch::kFloat32;
  } else {
      error->all(FLERR, "the model requested an unsupported dtype '{}'", mta_data->capabilities->dtype());
  }

  // transform from LAMMPS to metatomic System
  auto system = this->system_adaptor->system_from_lmp(
      mta_list,
      static_cast<bool>(vflag_global),
      mta_data->remap_pairs,
      dtype,
      mta_data->device
  );

  // gather masses (per-atom) in a tensor and ship to device
  auto float_tensor_options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
  torch::Tensor masses;
  if (rmass) {
      masses = torch::from_blob(
          rmass, {nall},
          float_tensor_options.requires_grad(false)
      ).to(mta_data->device);
  } else {
      // need to map from atom type to mass
      std::vector<double> masses_vector(nall);
      for (int i=0; i<nall; i++) {
          masses_vector[i] = mass[type[i]];
      }
      masses = torch::from_blob(
          masses_vector.data(), {nall},
          float_tensor_options.requires_grad(false)
      ).to(mta_data->device);
  }
  
  auto label_tensor_options = torch::TensorOptions().dtype(torch::kInt32).device(mta_data->device);
  // add masses to system
  {
    metatensor_torch::Labels keys = metatensor_torch::LabelsHolder::single()->to(mta_data->device);
    auto samples_tensor = torch::column_stack({
        torch::zeros(nall, label_tensor_options).unsqueeze(1),
        torch::arange(nall, label_tensor_options).unsqueeze(1)
    });
    metatensor_torch::Labels samples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
      std::vector<std::string>{"system","atom"}, samples_tensor);
    auto properties = metatensor_torch::LabelsHolder::single()->to(mta_data->device);
    auto block = torch::make_intrusive<metatensor_torch::TensorBlockHolder>(
      masses.to(torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(-1),  // add property dimension
      samples,
      std::vector<metatensor_torch::Labels>{},
      properties
    );
    auto blocks = std::vector<metatensor_torch::TensorBlock>{block};
    auto tmap = torch::make_intrusive<metatensor_torch::TensorMapHolder>(keys, blocks);
    system->add_data("masses", tmap, /*override=*/true);
  }

  // add momenta to the system
  {
    // gather velocities in a tensor and ship to device
    auto velocities = torch::from_blob(
        // atom->v contains "real" and then ghost atoms, in that order
        *v, {nall, 3},
        // since Metatomic is not a force field, there's no need to allocate space to store gradients
        float_tensor_options.requires_grad(false)
    ).to(mta_data->device);

    // compute momenta = mass * velocity
    auto momenta = masses.unsqueeze(1) * velocities * (0.001 / 0.09822694743391452);
    // std::cout << "Momenta before:" << std::endl;
    // std::cout << momenta.index({torch::indexing::Slice(0, nlocal), torch::indexing::Slice()}) << std::endl;
    // exit(0);
    // auto momenta = masses.unsqueeze(1) * velocities * 0.0;

    // print only the first n_local momenta (i.e. excluding ghosts)
    // std::cout << momenta.index({torch::indexing::Slice(0, nlocal), torch::indexing::Slice()}) << std::endl;

    // define TensorBlock
    auto keys = metatensor_torch::LabelsHolder::single()->to(mta_data->device);
    auto values = momenta.unsqueeze(-1); // add property dimension

    // define samples
    auto sample_value_components = std::vector<torch::Tensor>{
        torch::zeros(nall, label_tensor_options).unsqueeze(1),
        torch::arange(nall, label_tensor_options).unsqueeze(1)
    };
    auto sample_values = torch::column_stack(sample_value_components);
    metatensor_torch::Labels samples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
        std::vector<std::string>{"system", "atom"}, sample_values
    );

    // define components
    auto component_values = torch::arange(3, label_tensor_options).unsqueeze(1);
    metatensor_torch::Labels components = torch::make_intrusive<metatensor_torch::LabelsHolder>(
        std::vector<std::string>{"xyz"}, component_values
    );

    auto properties = metatensor_torch::LabelsHolder::single()->to(mta_data->device);
    auto block = torch::make_intrusive<metatensor_torch::TensorBlockHolder>(
      // TODO: is there a way to check what dtype the model expects for input data?
      values.to(torch::TensorOptions().dtype(torch::kFloat32)),
      samples,
      std::vector<metatensor_torch::Labels>{components},
      properties
    );
    auto blocks = std::vector<metatensor_torch::TensorBlock>{block};
    auto tmap = torch::make_intrusive<metatensor_torch::TensorMapHolder>(keys, blocks);
    system->add_data("momenta", tmap, /*override=*/true);
  }

  // only run the calculation for atoms actually in the current domain
  mta_data->selected_atoms_values.resize_({atom->nlocal, 2});
  mta_data->selected_atoms_values.index_put_({torch::indexing::Slice(), 0}, 0);
  auto options = mta_data->selected_atoms_values.options();
  mta_data->selected_atoms_values.index_put_(
      {torch::indexing::Slice(), 1},
      torch::arange(atom->nlocal, options)
  );

  auto selected_atoms = torch::make_intrusive<metatensor_torch::LabelsHolder>(
      std::vector<std::string>{"system", "atom"}, mta_data->selected_atoms_values
  );
  mta_data->evaluation_options->set_selected_atoms(selected_atoms);

  // std::cout << system->positions() << std::endl;
  // std::cout << metatensor_torch::TensorMapHolder::block_by_id(system->get_data("masses"), 0)->values() << std::endl;
  // std::cout << metatensor_torch::TensorMapHolder::block_by_id(system->get_data("momenta"), 0)->values().squeeze(-1) << std::endl;
  // exit(0);
  // std::cout << system->types() << std::endl;
  // std::cout << system->cell() << std::endl;
  // std::cout << system->pbc() << std::endl;

  // call the model to get delta-positions and updated momenta
  torch::IValue result_ivalue;
  try {
    // run the model
      result_ivalue = mta_data->model->forward({
          std::vector<metatomic_torch::System>{system},
          mta_data->evaluation_options,
          mta_data->check_consistency
      });
  } catch (const std::exception& e) {
      error->all(FLERR, "error evaluating the torch model: {}", e.what());
  }

  // apply the results to LAMMPS atoms
  auto result = result_ivalue.toGenericDict();

  // extract position updates
  auto positions_map = result.at("positions").toCustomClass<metatensor_torch::TensorMapHolder>();
  auto positions_block = metatensor_torch::TensorMapHolder::block_by_id(positions_map, 0);
  auto positions = positions_block->values().squeeze(-1).to(torch::kCPU).to(torch::kFloat64);

  // extract momenta updates
  auto momenta_map = result.at("momenta").toCustomClass<metatensor_torch::TensorMapHolder>();
  auto momenta_block = metatensor_torch::TensorMapHolder::block_by_id(momenta_map, 0);
  auto momenta = momenta_block->values().squeeze(-1).to(torch::kCPU).to(torch::kFloat64);

  // std::cout << positions << std::endl;
  // std::cout << momenta << std::endl;
  // exit(0);
  // jdgfkakjd

  // std::cout << momenta << std::endl;
  // exit(0);

  momenta = momenta / (0.001 / 0.09822694743391452);

  for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
          // update positions
          x[i][0] = positions[i][0].item<double>();
          x[i][1] = positions[i][1].item<double>();
          x[i][2] = positions[i][2].item<double>();

          // std::cout << "Before: " << v[i][0];

          // update velocities based on new momenta
          v[i][0] = momenta[i][0].item<double>() / masses[i].item<double>();
          v[i][1] = momenta[i][1].item<double>() / masses[i].item<double>();
          v[i][2] = momenta[i][2].item<double>() / masses[i].item<double>();

          // std::cout << " After: " << v[i][0] << std::endl;
      }
  }
}

void FixMetatomic::post_force(int /*vflag*/)
{
  // take a snapshot of forces
  this->ensure_capacity();

  double **f = atom->f;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++) {
    f_pre[i][0] = f[i][0];
    f_pre[i][1] = f[i][1];
    f_pre[i][2] = f[i][2];
  }
}

void FixMetatomic::final_integrate()
{
  double dtf = update->dt * force->ftm2v;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++) {
    v[i][0] += (f[i][0] - f_pre[i][0]) * dtf / (rmass ? rmass[i] : atom->mass[atom->type[i]]);
    v[i][1] += (f[i][1] - f_pre[i][1]) * dtf / (rmass ? rmass[i] : atom->mass[atom->type[i]]);
    v[i][2] += (f[i][2] - f_pre[i][2]) * dtf / (rmass ? rmass[i] : atom->mass[atom->type[i]]);
  }
}


void FixMetatomic::ensure_capacity()
{
  if (atom->nmax > nmax) {
    this->nmax = atom->nmax;
    if (f_pre) memory->destroy(f_pre);
    memory->create(f_pre, this->nmax, 3, "FixMetatomic::f_pre");
  }
}
