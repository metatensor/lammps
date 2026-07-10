/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS Development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Guillaume Fraux <guillaume.fraux@epfl.ch>
------------------------------------------------------------------------- */
#include "metatomic_types.h"

#include "citeme.h"
#include "comm.h"
#include "error.h"

#include <algorithm>
#include <cmath>

#include <torch/cuda.h>

using namespace LAMMPS_NS;

CommonMetatomicData::CommonMetatomicData(std::string length_unit): device(torch::kCPU) {
    auto options = torch::TensorOptions().dtype(torch::kInt32);
    this->selected_atoms_values = torch::zeros({0, 2}, options);
    this->selected_atoms_values_cpu = torch::zeros({0, 2}, options);

    // Initialize evaluation_options
    this->evaluation_options = torch::make_intrusive<metatomic_torch::ModelEvaluationOptionsHolder>();
    this->evaluation_options->set_length_unit(std::move(length_unit));
}

void CommonMetatomicData::load_model(
   LAMMPS* lmp,
   const char* path,
   const char* extensions_directory
) {
   // TODO: seach for the model & extensions inside `$LAMMPS_POTENTIALS`?

   this->model_path = path;
   if (this->model != nullptr) {
       lmp->error->one(FLERR, "torch model is already loaded");
   }

   torch::optional<std::string> extensions = torch::nullopt;
   if (extensions_directory != nullptr) {
       extensions = std::string(extensions_directory);
   }

   try {
       this->model = std::make_unique<metatensor_torch::Module>(
           metatomic_torch::load_atomistic_model(this->model_path, extensions)
       );
   } catch (const c10::Error& e) {
       lmp->error->one(FLERR, "failed to load metatomic model at '{}': {}", path, e.what());
   }

   auto capabilities_ivalue = this->model->run_method("capabilities");
   this->capabilities = capabilities_ivalue.toCustomClass<metatomic_torch::ModelCapabilitiesHolder>();

   if (lmp->comm->me == 0) {
       auto metadata_ivalue = this->model->run_method("metadata");
       auto metadata = metadata_ivalue.toCustomClass<metatomic_torch::ModelMetadataHolder>();
       auto to_print = metadata->print();

       if (lmp->screen) {
           fprintf(lmp->screen, "\n%s\n", to_print.c_str());
       }
       if (lmp->logfile) {
           fprintf(lmp->logfile,"\n%s\n", to_print.c_str());
       }

       // add the model references to LAMMPS citation handling mechanism
       if (lmp->citeme) {
          for (const auto& it: metadata->references) {
             for (const auto& ref: it.value()) {
                lmp->citeme->add(ref + "\n");
             }
          }
       }
   }
}

void CommonMetatomicData::pick_device(LAMMPS* lmp, const char* requested, const char* cmd_name) {
    torch::optional<std::string> requested_string;
    if (requested != nullptr) {
        requested_string = std::string(requested);
    } else {
        requested_string = torch::nullopt;
    }

    torch::DeviceType device_type;
    try {
        device_type = metatomic_torch::pick_device(
            this->capabilities->supported_devices,
            requested_string
        );
    } catch (const c10::Error& e) {
        lmp->error->one(FLERR, "{}: {}", cmd_name, e.what());
    }

    if (device_type == torch::DeviceType::CUDA) {
        // distribute GPUs between multiple MPI processes on the same node

        // (1) get a MPI communicator for all processes on the current node
        MPI_Comm local;
        MPI_Comm_split_type(lmp->world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
        // (2) get the rank of this MPI process on the current node
        int local_rank;
        MPI_Comm_rank(local, &local_rank);

        int size;
        MPI_Comm_size(local, &size);
        if (size < torch::cuda::device_count()) {
            if (lmp->comm->me == 0) {
                lmp->error->warning(FLERR,
                    "found {} CUDA-capable GPUs, but only {} MPI processes on the current node; the remaining GPUs will not be used",
                    torch::cuda::device_count(), size
                );
            }
        }

        // (3) split GPUs between node-local processes using round-robin allocation
        auto device_index = local_rank % torch::cuda::device_count();
        this->device = torch::Device(device_type, static_cast<torch::DeviceIndex>(device_index));
    } else {
        this->device = torch::Device(device_type);
    }
}

c10::ScalarType CommonMetatomicData::model_dtype(LAMMPS* lmp) const {
    const auto dtype = this->capabilities->dtype();
    if (dtype == "float64") {
        return torch::kFloat64;
    } else if (dtype == "float32") {
        return torch::kFloat32;
    } else {
        lmp->error->all(FLERR, "the model requested an unsupported dtype '{}'", dtype);
    }
    return torch::kFloat64;  // unreachable, error->all does not return
}

void CommonMetatomicData::resolve_max_cutoff(LAMMPS* lmp) {
    // get the model's interaction range
    auto range = this->capabilities->engine_interaction_range(this->evaluation_options->length_unit());
    if (range < 0) {
        lmp->error->all(FLERR, "interaction_range is negative for this model");
    } else if (!std::isfinite(range)) {
        if (lmp->comm->nprocs > 1) {
            lmp->error->all(FLERR,
                "interaction_range is infinite for this model, "
                "using multiple MPI domains is not supported"
            );
        }

        // determine the maximal cutoff in the NL
        auto requested_nl = this->model->run_method("requested_neighbor_lists");
        for (const auto& ivalue: requested_nl.toList()) {
            auto options = ivalue.get().toCustomClass<metatomic_torch::NeighborListOptionsHolder>();
            auto cutoff = options->engine_cutoff(this->evaluation_options->length_unit());
            this->max_cutoff = std::max(this->max_cutoff, cutoff);
        }
    } else {
        this->max_cutoff = range;
    }
}

std::map<std::string, metatomic_torch::ModelOutput> CommonMetatomicData::collect_requested_inputs() const {
    std::map<std::string, metatomic_torch::ModelOutput> input_holders;
    auto requested_inputs = this->model->run_method("requested_inputs", /*use_new_names=*/ true).toGenericDict();
    for (const auto& entry : requested_inputs) {
        input_holders.emplace(
            entry.key().toStringRef(),
            entry.value().toCustomClass<metatomic_torch::ModelOutputHolder>()
        );
    }
    return input_holders;
}
