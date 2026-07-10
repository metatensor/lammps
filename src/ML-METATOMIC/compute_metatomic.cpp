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

#include "metatomic_types.h"
#include "metatomic_system.h"
#include "metatomic_quantities.h"

#include "compute_metatomic.h"

#include "atom.h"
#include "memory.h"
#include "modify.h"
#include "error.h"
#include "group.h"
#include "force.h"
#include "update.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "tokenizer.h"
#include "comm.h"

#include <vector>
#include <algorithm>

#include <metatomic/torch.hpp>
#include <metatensor/torch.hpp>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
ComputeMetatomic::ComputeMetatomic(LAMMPS *lmp, int narg, char **arg): Compute(lmp, narg, arg) {
    scalar_flag = 0;
    vector_flag = 0;
    peratom_flag = 0;
    size_vector_variable = 0;
    extscalar = 1;  // for now, we need to improve this in the future
    extvector = 1;

    std::string length_unit;
    if (strcmp(update->unit_style, "lj") == 0) {
        error->all(FLERR, "unsupported units '{}' for compute metatomic", update->unit_style);
    }
    length_unit = metatomic_unit_map.at("position").at(update->unit_style);

    if (narg < 5) {
        error->all(FLERR,
            "Illegal compute metatomic command: expected at least 5 arguments "
            "(compute ID group-ID metatomic model_path output_name ...); got %d", narg
        );
    }

    this->mta_data = new ComputeMetatomicData(std::move(length_unit));
    this->mta_data->check_consistency = false; // default, can be changed by the user
    this->model_path = arg[3];
    this->requested_device = std::nullopt;
    this->extensions_directory = std::nullopt;
    this->output_name = arg[4];

    int iarg = 5;

    bool unit_is_set = false;
    bool per_atom_is_set = false;
    bool shape_is_set = false;
    bool output_size_is_set = false;
    bool types_are_set = false;
    std::string unit;
    bool per_atom;
    std::string shape;
    int output_size = 0;
    std::vector<int> parsed_types;

    // let's decide if the user should specify the unit and the shape
    if (this->output_name.find("::") == std::string::npos) {
        // standard output, but may come with a variant ("quantity/variant")
        auto output_quantity_name = Tokenizer(this->output_name, "/").as_vector()[0];
        auto it = metatomic_unit_map.find(output_quantity_name);
        if (it != metatomic_unit_map.end()) {
            unit = it->second.at(update->unit_style);
            unit_is_set = true;
        } else if (strcmp(output_quantity_name.c_str(), "feature") == 0) {
            // features are dimensionless, no unit conversion
            unit = "";
            unit_is_set = true;
        } else {
            error->all(FLERR,
                "the output quantity '{}' is not a standard metatomic quantity, "
                "or have not been registered in the `metatomic_unit_map` map, "
                "if it is the former case, the quantity should be named as "
                "'<domain>::<quantity>'; if it is the latter case, please contact the "
                "developers to register the quantity '{}' in the"
                "`metatomic_unit_map` map",
                output_quantity_name
            );
        }
        auto it2 = metatomic_quantity_shape.find(output_quantity_name);
        if (it2 != metatomic_quantity_shape.end()) {
            shape = it2->second.at("shape");
            shape_is_set = true;
            if (shape == "vector") {
                output_size = std::stoi(it2->second.at("size"));
                output_size_is_set = true;
            }
        } else if (strcmp(output_quantity_name.c_str(), "feature") == 0) {
            // we know nothing about features, the user must specify the shape and size
        } else {
            error->all(FLERR,
                "the output quantity '{}' is not a standard metatomic quantity, "
                "or have not been registered in the `metatomic_quantity_shape` map, "
                "if it is the former case, the quantity should be named as "
                "'<domain>::<quantity>'; if it is the latter case, please contact the "
                "developers to register the quantity '{}' in the"
                "`metatomic_quantity_shape` map",
                output_quantity_name
            );
        }

    } // else: non-standard output, user must specify unit; handled by the !unit_is_set check below

    while (iarg < narg) {
        if (strcmp(arg[iarg], "unit") == 0) {
            iarg += 1;
            if (iarg == narg) {
                error->one(FLERR, "expected <unit_name> after 'unit' in compute metatomic, got nothing");
            } else {
                unit = std::string(arg[iarg]);
                unit_is_set = true;
                iarg += 1;
            }
        } else if (strcmp(arg[iarg], "device") == 0) {
            iarg += 1;
            if (iarg == narg) {
                error->one(FLERR, "Illegal compute metatomic command: 'device' expects an argument "
                    "specifying the device (e.g. cpu, cuda, mps)");
            }
            this->requested_device = std::string(arg[iarg]);
            iarg += 1;
        } else if (strcmp(arg[iarg], "check_consistency") == 0) {
            iarg += 1;
            if (iarg == narg) {
                error->one(FLERR, "expected <on/off> after 'check_consistency' in compute metatomic, got nothing");
            } else if (strcmp(arg[iarg], "on") == 0) {
                mta_data->check_consistency = true;
                iarg += 1;
            } else if (strcmp(arg[iarg], "off") == 0) {
                mta_data->check_consistency = false;
                iarg += 1;
            } else {
                error->one(FLERR, "expected <on/off> after 'check_consistency' in compute metatomic, got '{}'", arg[iarg]);
            }
        } else if (strcmp(arg[iarg], "extensions_directory") == 0) {
            iarg += 1;
            if (iarg == narg) {
                error->one(FLERR, "expected <directory_path> after 'extensions_directory' in compute metatomic, got nothing");
            } else {
                this->extensions_directory = std::string(arg[iarg]);
                iarg += 1;
            }
        } else if (strcmp(arg[iarg], "shape") == 0) {
            iarg += 1;
            if (iarg == narg) {
                error->one(FLERR, "expected <scalar/vector> after 'shape' in compute metatomic, got nothing");
            } else if (strcmp(arg[iarg], "scalar") == 0) {
                shape = arg[iarg];
                shape_is_set = true;
                iarg += 1;
            } else if (strcmp(arg[iarg], "vector") == 0) {
                shape = arg[iarg];
                shape_is_set = true;
                iarg += 1;
                if (iarg == narg) {
                    error->one(FLERR, "expected <size> after 'vector' in compute metatomic, got nothing");
                }
                output_size = std::stoi(arg[iarg]);
                output_size_is_set = true;
                iarg += 1;
            } else {
                error->one(FLERR, "expected <scalar/vector> after 'shape' in compute metatomic, got '{}'", arg[iarg]);
            }
        } else if (strcmp(arg[iarg], "types") == 0) {
            iarg += 1;
            types_are_set = true;
            // Require exactly atom->ntypes integer values after the "types" keyword.
            if (iarg + atom->ntypes > narg) {
                error->all(FLERR,
                    "Illegal compute metatomic command: expected {} type values "
                    "after 'types'", atom->ntypes
                );
            }
            for (int ti = 0; ti < atom->ntypes; ++ti) {
                int type = -1;
                const char *argstr = arg[iarg + ti];
                try {
                    type = std::stoi(argstr);
                } catch (const std::invalid_argument &) {
                    error->all(FLERR,
                        "Illegal compute metatomic command: expected integer for type {}, "
                        "got '{}'", ti + 1, argstr
                    );
                } catch (const std::out_of_range &) {
                    error->all(FLERR,
                        "Illegal compute metatomic command: type value out of range "
                        "for argument '%s'", argstr
                    );
                }
                if (type <= 0) {
                    error->all(FLERR, "Illegal compute metatomic command: type {} should be > 0", type);
                }
                parsed_types.push_back(type);
            }
            iarg += atom->ntypes;
        } else {
            error->all(FLERR,
            "Illegal compute metatomic command: unrecognized keyword '{}' (expected 'unit', 'device', 'check_consistency', 'shape', or 'types' )", arg[iarg]);
        }
    }

    if (!unit_is_set) {
        error->all(FLERR, "Illegal compute metatomic command: 'unit' keyword is required");
    }
    if (!types_are_set) {
        error->all(FLERR, "Illegal compute metatomic command: no types specified");
    }
    if (!shape_is_set) {
        error->all(FLERR, "Illegal compute metatomic command: 'shape' keyword is required");
    }
    if (strcmp(shape.c_str(), "vector") == 0 && !output_size_is_set) {
        error->all(FLERR, "Illegal compute metatomic command: 'size' keyword is required when 'shape' is 'vector'");
    }

    // Allocate and fill the type-mapping (1-based indexing)
    type_mapping = memory->create(type_mapping, atom->ntypes + 1, "compute_metatomic:type_mapping");
    for (int i = 1; i <= atom->ntypes; i++) {
        type_mapping[i] = parsed_types[i - 1];
    }

    mta_data->load_model(
        this->lmp,
        this->model_path.c_str(),
        this->extensions_directory ? this->extensions_directory->c_str() : nullptr
    );

    auto capabilities = mta_data->model->run_method("capabilities").toCustomClass<metatomic_torch::ModelCapabilitiesHolder>();
    c10::ScalarType dtype;
    if (capabilities->dtype() == "float64") {
        dtype = torch::kFloat64;
    } else if (capabilities->dtype() == "float32") {
        dtype = torch::kFloat32;
    } else {
        error->all(FLERR,
            "the model requested an unsupported dtype '" + capabilities->dtype() + "'"
        );
    }
    auto model_outputs = capabilities->outputs();
    if (!model_outputs.contains(this->output_name)) {
        error->all(FLERR,
            "the model does not provide an output named '" + this->output_name + "'"
        );
    }
    auto sample_kind = model_outputs.at(this->output_name)->sample_kind();
    this->mta_data->requested_output = torch::make_intrusive<metatomic_torch::ModelOutputHolder>(
        /*quantity=*/"",
        /*unit=*/unit,
        /*sample_kind=*/sample_kind,
        /*explicit_gradients=*/std::vector<std::string>{},
        /*description=*/"Requested output from LAMMPS compute metatomic"
    );
    this->mta_data->evaluation_options->outputs.insert(this->output_name, this->mta_data->requested_output);

    // add the required additional inputs
    auto requested_inputs = mta_data->model->run_method("requested_inputs", /*use_new_names=*/ true).toGenericDict();
    for (const auto& entry : requested_inputs) {
        mta_data->requested_inputs.emplace(
            entry.key().toStringRef(),
            entry.value().toCustomClass<metatomic_torch::ModelOutputHolder>()
        );
    }

    // Initialize the output layout
    if (strcmp(sample_kind.c_str(), "atom") == 0) {
        peratom_flag = 1;
        if (strcmp(shape.c_str(), "scalar") == 0) {
            result_kind = RESULT_PERATOM_SCALAR;
            size_peratom_cols = 0;
        } else if (strcmp(shape.c_str(), "vector") == 0) {
            result_kind = RESULT_PERATOM_VECTOR;
            size_peratom_cols = output_size;
        } else {
            error->all(FLERR, "Illegal compute metatomic command: 'shape' must be 'scalar' or 'vector'");
        }
    } else if (strcmp(sample_kind.c_str(), "system") == 0) {
        if (strcmp(shape.c_str(), "scalar") == 0) {
            result_kind = RESULT_GLOBAL_SCALAR;
            scalar_flag = 1;
        } else if (strcmp(shape.c_str(), "vector") == 0) {
            result_kind = RESULT_GLOBAL_VECTOR;
            vector_flag = 1;
            size_vector = output_size;
        } else {
            error->all(FLERR, "Illegal compute metatomic command: 'shape' must be 'scalar' or 'vector'");
        }
    } else {
        error->all(FLERR, "The requested output '" + this->output_name + "' has an unsupported sample kind '" + sample_kind + "'");
    }


    // Select the device to use based on the model's preference, the user choice
    // and what's available.
    this->pick_device(
        mta_data->device,
        this->requested_device ? this->requested_device->c_str() : nullptr
    );

    // move all data to the correct device
    mta_data->model->to(mta_data->device);
    mta_data->selected_atoms_values = mta_data->selected_atoms_values.to(mta_data->device);
}

ComputeMetatomic::~ComputeMetatomic() {
    memory->destroy(type_mapping);
    delete mta_data;
    if (result_kind == RESULT_GLOBAL_VECTOR) {
        memory->destroy(vector);
    } else if (result_kind == RESULT_PERATOM_SCALAR) {
        memory->destroy(vector_atom);
    } else if (result_kind == RESULT_PERATOM_VECTOR) {
        memory->destroy(array_atom);
    } else if (result_kind == RESULT_GLOBAL_SCALAR || result_kind == RESULT_NONE) {
        // nothing to destroy
    } else {
        error->all(FLERR, "compute metatomic internal error: unknown result kind in destructor");
    }
}

void ComputeMetatomic::init() {
    auto message = "Computing " + this->output_name + " on " + mta_data->device.str() + " device with " + mta_data->capabilities->dtype() + " data";
    if (screen) {
        fprintf(screen, "%s\n", message.c_str());
    }
    if (logfile) {
        fprintf(logfile,"%s\n", message.c_str());
    }

    if (!type_mapping) {
        error->all(FLERR, "compute metatomic internal error: type_mapping not initialized");
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
        /* requires_grad */ false,
    };
    this->system_adaptor = std::make_unique<MetatomicSystemAdaptor>(lmp, options);

    // We ask LAMMPS for a full neighbor lists because we need to know about
    // ALL pairs, even if options->full_list() is false. We will then filter
    // the pairs to only include each pair once where needed.
    auto request = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
    request->set_cutoff(mta_data->max_cutoff);

    auto mincut = mta_data->max_cutoff + neighbor->skin;
    if (comm->get_comm_cutoff() < mincut) {
        if (comm->me == 0) {
            error->warning(FLERR,
                "Increasing communication cutoff to {:.8} for compute metatomic",
                mincut
            );
        }
        comm->cutghostuser = mincut;
    }

    // Translate from the metatomic neighbor lists requests to LAMMPS neighbor
    // lists requests.
    auto requested_nl = mta_data->model->run_method("requested_neighbor_lists");
    for (const auto& ivalue: requested_nl.toList()) {
        auto options = ivalue.get().toCustomClass<metatomic_torch::NeighborListOptionsHolder>();
        auto cutoff = options->engine_cutoff(mta_data->evaluation_options->length_unit());
        assert(cutoff <= mta_data->max_cutoff);

        this->system_adaptor->add_nl_request(cutoff, options);
    }

    // HACK: Explicitly set the binsize for the neighbor list if there is no
    // pair_style that would set it instead.
    //
    // Otherwise, the default binsize of box[0] is used, which crashes kokkos
    // for large-ish boxes (~40A), and slow down the simulation for non-kokkos.
    if (strcmp(force->pair_style, "none") == 0) {
        neighbor->binsize_user = 0.5 * mta_data->max_cutoff;
        neighbor->binsizeflag = 1;
    }
    // END HACK
}

void ComputeMetatomic::pick_device(c10::Device& device, const char* requested) {
    torch::optional<std::string> requested_string;
    torch::DeviceType device_type;

    if (requested != nullptr) {
        requested_string = std::string(requested);
    } else {
        requested_string = torch::nullopt;
    }

    try {
        device_type = metatomic_torch::pick_device(
            this->mta_data->capabilities->supported_devices,
            requested_string
        );
    } catch (const c10::Error& e) {
        error->one(FLERR, "compute metatomic: {}", e.what());
    }

    if (device_type == torch::DeviceType::CUDA) {
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
        auto device_index = local_rank % torch::cuda::device_count();
        device = torch::Device(device_type, static_cast<torch::DeviceIndex>(device_index));
    } else {
        device = torch::Device(device_type);
    }
}

void ComputeMetatomic::init_list(int id, NeighList *ptr) {
    mta_list = ptr;
}

void ComputeMetatomic::compute() {
    if (update->ntimestep == last_eval_step) {
        // already computed the model output for the current configuration, no need to do it again
        return;
    }
    clear_cache();
    int nlocal = atom->nlocal;
    int *mask = atom->mask;

    // Determine the dtype of the system based on the model's capabilities
    auto dtype = torch::kFloat64;
    if (mta_data->capabilities->dtype() == "float64") {
        dtype = torch::kFloat64;
    } else if (mta_data->capabilities->dtype() == "float32") {
        dtype = torch::kFloat32;
    } else {
        error->all(FLERR, "the model requested an unsupported dtype '{}'", mta_data->capabilities->dtype());
    }

    auto system = this->system_adaptor->system_from_lmp(
        mta_list,
        false,
        dtype,
        mta_data->device,
        mta_data->requested_inputs
    );

    // Configure selected atoms for evaluation
    // Only run the calculation for atoms in the current group
    mta_data->selected_atoms_values.resize_({group->count(igroup), 2});
    mta_data->selected_atoms_values.index_put_({torch::indexing::Slice(), 0}, 0);
    int64_t idx = 0;
    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            mta_data->selected_atoms_values.index_put_({idx, 1}, i);
            idx++;
        }
    }

    auto selected_atoms = torch::make_intrusive<metatensor_torch::LabelsHolder>(
        std::vector<std::string>{"system", "atom"}, mta_data->selected_atoms_values
    );
    mta_data->evaluation_options->set_selected_atoms(selected_atoms);

    // Call the ML model to predict the requested output
    torch::IValue result_ivalue;
    try {
        result_ivalue = mta_data->model->forward({
            std::vector<metatomic_torch::System>{system},
            mta_data->evaluation_options,
            mta_data->check_consistency
        });
    } catch (const std::exception& e) {
        error->all(FLERR, "error evaluating the torch model: {}", e.what());
    }

    // Extract results from the model output
    auto result = result_ivalue.toGenericDict();

    // Extract requested output
    auto output_map = result.at(this->output_name).toCustomClass<metatensor_torch::TensorMapHolder>();
    auto output_block = metatensor_torch::TensorMapHolder::block_by_id(output_map, 0);
    auto output_values = output_block->values().to(torch::kCPU).to(torch::kFloat64).contiguous();

    auto output_samples = output_block->samples();
    auto output_samples_values = output_samples->values().to(torch::kCPU);
    auto output_samples_accessor = output_samples_values.accessor<int32_t, 2>();
    auto output_components = output_block->components();
    if (output_components.size() > 1) {
        error->all(FLERR, "compute metatomic currently only supports outputs with at most 1 component dimension, got {}", output_components.size());
    }

    ResultKind kind = RESULT_NONE;
    n_samples = output_values.size(0);
    n_properties = output_values.size(output_values.dim() - 1);
    n_components = output_values.dim() == 3 ? output_values.size(1) : 1;
    if (n_properties != 1) {
        error->all(FLERR, "compute metatomic currently only supports outputs with a one property, got {}", n_properties);
    }

    if (output_samples->size() == 1) {
        assert (output_samples->names()[0] == "system");
        if (output_components.size() == 1) {
            // a system-level output, having components means it's a vector quantity (e.g. heat flux)
            kind = RESULT_GLOBAL_VECTOR;
            vector_cache_ = std::vector<double>(output_values.data_ptr<double>(), output_values.data_ptr<double>() + output_values.numel());
        } else if (output_components.size() == 0) {
            // a system-level output with no components is a scalar quantity (e.g. potential energy)
            kind = RESULT_GLOBAL_SCALAR;
            scalar_cache_ = output_values.squeeze().item<double>();
        } else {
            error->all(FLERR, "compute metatomic: expected at most 1 component label for global outputs, got {}", output_components.size());
        }
    } else if (output_samples->size() == 2) {
        assert (output_samples->names()[0] == "system");
        assert (output_samples->names()[1] == "atom");
        // map from metatomic atom indices to local LAMMPS atom indices
        const auto& mta_to_lmp = this->system_adaptor->mta_to_lmp;
        local_indices_cache_ = {};
        for (int i = 0; i < n_samples; i++) {
            assert(output_samples_accessor[i][0] == 0);
            // handle potentially out of order samples in
            // the per-atom energy tensor
            auto atom_i = mta_to_lmp[output_samples_accessor[i][1]];
            local_indices_cache_.push_back(atom_i);
        }
        if (output_components.size() == 1) {
            // a per-atom vector quantity (e.g. forces)
            kind = RESULT_PERATOM_VECTOR;
            peratom_array_cache_ = std::vector<double>(output_values.data_ptr<double>(), output_values.data_ptr<double>() + output_values.numel());
        } else if (output_components.size() == 0) {
            // a per-atom scalar quantity (e.g. atomic energy)
            kind = RESULT_PERATOM_SCALAR;
            peratom_cache_ = std::vector<double>(output_values.data_ptr<double>(), output_values.data_ptr<double>() + output_values.numel());
        } else {
            error->all(FLERR, "compute metatomic: expected at most 1 component label for per-atom outputs, got {}", output_components.size());
        }
    } else {
        error->all(FLERR, "compute metatomic: expected output samples to have either 1 or 2 labels, got {}", output_samples->size());
    }

    check_or_init_layout(kind, n_components, n_components != 1 ? n_components : 0);
    last_eval_step = update->ntimestep;
}

double ComputeMetatomic::compute_scalar() {
    invoked_scalar = update->ntimestep;
    this->compute();
    if (result_kind != RESULT_GLOBAL_SCALAR) {
        error->all(FLERR, "compute metatomic: compute_scalar called but the output is not a scalar");
    }
    scalar = scalar_cache_;
    return scalar;
}

void ComputeMetatomic::compute_vector() {
    invoked_vector = update->ntimestep;
    this->compute();
    if (result_kind != RESULT_GLOBAL_VECTOR) {
        error->all(FLERR, "compute metatomic: compute_vector called but the output is not a vector");
    }
    if (vector_cache_.size() != n_components) {
        error->all(FLERR, "compute metatomic: expected vector output of size {}, got {}", n_components, vector_cache_.size());
    }
    for (int i = 0; i < n_components; i++) {
        vector[i] = vector_cache_[i];
    }
}

void ComputeMetatomic::compute_peratom() {
    invoked_peratom = update->ntimestep;
    this->compute();
    if (result_kind == RESULT_PERATOM_SCALAR) {
        if (peratom_cache_.size() != n_samples) {
            error->all(FLERR, "compute metatomic: expected per-atom output of size {}, got {}", n_samples, peratom_cache_.size());
        }
        for (int i = 0; i < n_max_atoms; i++) {
            vector_atom[i] = 0.0;
        }
        for (int i = 0; i < n_samples; i++) {
            vector_atom[local_indices_cache_[i]] = peratom_cache_[i];
        }
    } else if (result_kind == RESULT_PERATOM_VECTOR) {
        if (peratom_array_cache_.size() != n_samples * n_components) {
            error->all(FLERR, "compute metatomic: expected per-atom array output of size {}, got {}", n_samples * n_components, peratom_array_cache_.size());
        }
        for (int i = 0; i < n_max_atoms; i++) {
            for (int j = 0; j < n_components; j++) {
                array_atom[i][j] = 0.0;
            }
        }
        for (int i = 0; i < n_samples; i++) {
            for (int j = 0; j < n_components; j++) {
                array_atom[local_indices_cache_[i]][j] = peratom_array_cache_[i * n_components + j];
            }
        }
    } else {
        error->all(FLERR, "compute metatomic: compute_peratom called but the output is not per-atom");
    }
}


void ComputeMetatomic::check_or_init_layout(ResultKind kind, int size, int peratom_cols) {
    if (!layout_initialized) {
        // check that the user-specified output layout matches the model output, if so initialize the
        layout_initialized = true;
        n_max_atoms = atom->nmax;
        if (result_kind != kind) {
            error->all(FLERR, "compute metatomic: expected user-specified output kind {}, got {} from model", result_kind_name(result_kind), result_kind_name(kind));
        }
        if (result_kind == RESULT_GLOBAL_VECTOR) {
            if (size_vector != size) {
                error->all(FLERR, "compute metatomic: expected user-specified output of size {}, got {}", size_vector, size);
            }
            memory->create(vector, size_vector, "compute_metatomic:vector");
        }
        if (result_kind == RESULT_PERATOM_SCALAR) {
            if (size_peratom_cols != peratom_cols) {
                error->all(FLERR, "compute metatomic: expected per-atom scalar output, got a non-scalar output with {} components", size_peratom_cols, peratom_cols);
            }
            memory->create(vector_atom, n_max_atoms, "compute_metatomic:vector_atom");
        }
        if (result_kind == RESULT_PERATOM_VECTOR) {
            if (size_peratom_cols != peratom_cols) {
                error->all(FLERR, "compute metatomic: expected per-atom vector output with {} columns, got {}", size_peratom_cols, peratom_cols);
            }
            memory->create(array_atom, n_max_atoms, size_peratom_cols, "compute_metatomic:array_atom");
        }
    } else {
        if (result_kind != kind) {
            error->all(FLERR, "compute metatomic: expected output of kind {}, got {}", result_kind_name(result_kind), result_kind_name(kind));
        }
        if (result_kind == RESULT_GLOBAL_VECTOR && size_vector != size) {
            error->all(FLERR, "compute metatomic: expected output vector of size {}, got {}", size_vector, size);
        }
        if ((result_kind == RESULT_PERATOM_SCALAR || result_kind == RESULT_PERATOM_VECTOR) && size_peratom_cols != peratom_cols) {
            error->all(FLERR, "compute metatomic: expected per-atom array output with {} columns, got {}", size_peratom_cols, peratom_cols);
        }
        if (n_max_atoms < atom->nmax) {
            n_max_atoms = atom->nmax;
            if (result_kind == RESULT_PERATOM_SCALAR) {
                memory->destroy(vector_atom);
                memory->create(vector_atom, n_max_atoms, "compute_metatomic:vector_atom");
            }
            if (result_kind == RESULT_PERATOM_VECTOR) {
                memory->destroy(array_atom);
                memory->create(array_atom, n_max_atoms, size_peratom_cols, "compute_metatomic:array_atom");
            }
        }
    }
}

const char *ComputeMetatomic::result_kind_name(ResultKind kind)
{
  switch (kind) {
    case RESULT_NONE: return "none";
    case RESULT_GLOBAL_SCALAR: return "scalar";
    case RESULT_GLOBAL_VECTOR: return "global_vector";
    case RESULT_PERATOM_SCALAR: return "peratom_vector";
    case RESULT_PERATOM_VECTOR: return "peratom_array";
  }
  return "unknown";
}

void ComputeMetatomic::clear_cache() {
    scalar_cache_ = 0.0;
    vector_cache_.clear();
    peratom_cache_.clear();
    peratom_array_cache_.clear();
    local_indices_cache_.clear();
    n_samples = 0;
};
