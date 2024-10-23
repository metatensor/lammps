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
#include "pair_metatensor_kokkos.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "update.h"
#include "citeme.h"
#include "comm.h"

#include "neigh_list.h"

#include "kokkos.h"
#include "atom_kokkos.h"
#include "pair_kokkos.h"
#include "atom_masks.h"

#include <torch/version.h>
#include <torch/script.h>
#include <torch/cuda.h>

#if TORCH_VERSION_MAJOR >= 2
    #include <torch/mps.h>
#endif

#include <memory>

#include <metatensor/torch.hpp>
#include <metatensor/torch/atomistic.hpp>

#include "metatensor_system_kokkos.h"

#include <chrono>

#ifndef KOKKOS_ENABLE_CUDA
namespace Kokkos {
class Cuda {};
} // namespace Kokkos
#endif // KOKKOS_ENABLE_CUDA

using namespace LAMMPS_NS;

struct LAMMPS_NS::PairMetatensorDataKokkos {
    PairMetatensorDataKokkos(std::string length_unit, std::string energy_unit);

    void load_model(LAMMPS* lmp, const char* path, const char* extensions_directory);

    // torch model in metatensor format
    std::unique_ptr<torch::jit::Module> model;
    // device to use for the calculations
    torch::Device device;
    // model capabilities, declared by the model
    metatensor_torch::ModelCapabilities capabilities;
    // run-time evaluation options, decided by this class
    metatensor_torch::ModelEvaluationOptions evaluation_options;
    // should metatensor check the data LAMMPS send to the model
    // and the data the model returns?
    bool check_consistency;
    // whether pairs should be remapped, removing pairs between ghosts if there
    // is an equivalent pair involving at least one local atom.
    bool remap_pairs;
    // how far away the model needs to know about neighbors
    double max_cutoff;

    // adaptor from LAMMPS system to metatensor's
    std::unique_ptr<MetatensorSystemAdaptorKokkos<LMPDeviceType>> system_adaptor;
};

PairMetatensorDataKokkos::PairMetatensorDataKokkos(std::string length_unit, std::string energy_unit):
    system_adaptor(nullptr),
    device(torch::kCPU),
    check_consistency(false),
    remap_pairs(true),
    max_cutoff(-1)
{
    // default to true for now, this will be changed to false later
    this->check_consistency = true;

    // Initialize evaluation_options
    this->evaluation_options = torch::make_intrusive<metatensor_torch::ModelEvaluationOptionsHolder>();
    this->evaluation_options->set_length_unit(std::move(length_unit));

    auto output = torch::make_intrusive<metatensor_torch::ModelOutputHolder>();
    output->explicit_gradients = {};
    output->set_quantity("energy");
    output->set_unit(std::move(energy_unit));
    output->per_atom = false;

    this->evaluation_options->outputs.insert("energy", output);
}

void PairMetatensorDataKokkos::load_model(
    LAMMPS* lmp,
    const char* path,
    const char* extensions_directory
) {
    // TODO: seach for the model & extensions inside `$LAMMPS_POTENTIALS`?

    if (this->model != nullptr) {
        lmp->error->all(FLERR, "torch model is already loaded");
    }

    torch::optional<std::string> extensions = torch::nullopt;
    if (extensions_directory != nullptr) {
        extensions = std::string(extensions_directory);
    }

    try {
        this->model = std::make_unique<torch::jit::Module>(
            metatensor_torch::load_atomistic_model(path, extensions)
        );
    } catch (const c10::Error& e) {
        lmp->error->all(FLERR, "failed to load metatensor model at '{}': {}", path, e.what());
    }

    auto capabilities_ivalue = this->model->run_method("capabilities");
    this->capabilities = capabilities_ivalue.toCustomClass<metatensor_torch::ModelCapabilitiesHolder>();

    if (!this->capabilities->outputs().contains("energy")) {
        lmp->error->all(FLERR, "the model at '{}' does not have an \"energy\" output, we can not use it in pair_style metatensor", path);
    }

    if (lmp->comm->me == 0) {
        auto metadata_ivalue = this->model->run_method("metadata");
        auto metadata = metadata_ivalue.toCustomClass<metatensor_torch::ModelMetadataHolder>();
        auto to_print = metadata->print();

        if (lmp->screen) {
            fprintf(lmp->screen, "\n%s\n", to_print.c_str());
        }
        if (lmp->logfile) {
            fprintf(lmp->logfile,"\n%s\n", to_print.c_str());
        }

        // add the model references to LAMMPS citation handling mechanism
        for (const auto& it: metadata->references) {
            for (const auto& ref: it.value()) {
                lmp->citeme->add(ref + "\n");
            }
        }
    }
}


/* ---------------------------------------------------------------------- */

template<class LMPDeviceType>
PairMetatensorKokkos<LMPDeviceType>::PairMetatensorKokkos(LAMMPS *lmp): Pair(lmp), type_mapping(nullptr) {
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
        error->all(FLERR, "unsupported units '{}' for pair metatensor ", update->unit_style);
    }

    // we might not be running a pure pair potential,
    // so we can not compute virial as fdotr
    this->no_virial_fdotr_compute = 1;

    this->mts_data = new PairMetatensorDataKokkos(std::move(length_unit), std::move(energy_unit));
}

template<class LMPDeviceType>
PairMetatensorKokkos<LMPDeviceType>::~PairMetatensorKokkos() {
    delete this->mts_data;

    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(cutsq);
        memory->destroy(type_mapping);
    }
}

// called when finding `pair_style metatensor` in the input
template<class LMPDeviceType>
void PairMetatensorKokkos<LMPDeviceType>::settings(int argc, char ** argv) {
    if (argc == 0) {
        error->all(FLERR, "expected at least 1 argument to pair_style metatensor, got {}", argc);
    }

    const char* model_path = argv[0];
    const char* extensions_directory = nullptr;
    const char* requested_device = nullptr;
    for (int i=1; i<argc; i++) {
        if (strcmp(argv[i], "check_consistency") == 0) {
            if (i == argc - 1) {
                error->all(FLERR, "expected <on/off> after 'check_consistency' in pair_style metatensor, got nothing");
            } else if (strcmp(argv[i + 1], "on") == 0) {
                mts_data->check_consistency = true;
            } else if (strcmp(argv[i + 1], "off") == 0) {
                mts_data->check_consistency = false;
            } else {
                error->all(FLERR, "expected <on/off> after 'check_consistency' in pair_style metatensor, got '{}'", argv[i + 1]);
            }

            i += 1;
        } else if (strcmp(argv[i], "remap_pairs") == 0) {
            if (i == argc - 1) {
                error->all(FLERR, "expected <on/off> after 'remap_pairs' in pair_style metatensor, got nothing");
            } else if (strcmp(argv[i + 1], "on") == 0) {
                mts_data->remap_pairs = true;
            } else if (strcmp(argv[i + 1], "off") == 0) {
                mts_data->remap_pairs = false;
            } else {
                error->all(FLERR, "expected <on/off> after 'remap_pairs' in pair_style metatensor, got '{}'", argv[i + 1]);
            }

            i += 1;
        } else if (strcmp(argv[i], "extensions") == 0) {
            if (i == argc - 1) {
                error->all(FLERR, "expected <path> after 'extensions' in pair_style metatensor, got nothing");
            }
            extensions_directory = argv[i + 1];
            i += 1;
        } else if (strcmp(argv[i], "device") == 0) {
            if (i == argc - 1) {
                error->all(FLERR, "expected string after 'device' in pair_style metatensor, got nothing");
            }
            requested_device = argv[i + 1];
            i += 1;
        } else {
            error->all(FLERR, "unexpected argument to pair_style metatensor: '{}'", argv[i]);
        }
    }

    mts_data->load_model(this->lmp, model_path, extensions_directory);

    // Select the device to use based on the model's preference, the user choice
    // and what's available.
    auto available_devices = std::vector<torch::Device>();
    for (const auto& device: mts_data->capabilities->supported_devices) {
        if (device == "cpu") {
            available_devices.push_back(torch::kCPU);
        } else if (device == "cuda") {
            if (torch::cuda::is_available()) {
                // Get a MPI communicator for all processes on the current node
                MPI_Comm local;
                MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
                // Get the rank of this MPI process on the current node
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

                // split GPUs between node-local processes using round-robin allocation
                int gpu_to_use = local_rank % torch::cuda::device_count();
                available_devices.push_back(torch::Device(torch::kCUDA, gpu_to_use));
            }
        } else if (device == "mps") {
            #if TORCH_VERSION_MAJOR >= 2
            if (torch::mps::is_available()) {
                available_devices.push_back(torch::Device("mps"));
            }
            #endif
        } else {
            error->warning(FLERR,
                "the model declared support for unknown device '{}', it will be ignored", device
            );
        }
    }

    if (available_devices.empty()) {
        error->all(FLERR,
            "failed to find a valid device for the model at '{}': "
            "the model supports {}, none of these where available",  /// typo: where -> were
            model_path, torch::str(mts_data->capabilities->supported_devices)
        );
    }

    if (requested_device == nullptr) {
        // no user request, pick the device the model prefers
        mts_data->device = available_devices[0];
    } else {
        bool found_requested_device = false;
        for (const auto& device: available_devices) {
            if (device.is_cpu() && strcmp(requested_device, "cpu") == 0) {
                mts_data->device = device;
                found_requested_device = true;
                break;
            } else if (device.is_cuda() && strcmp(requested_device, "cuda") == 0) {
                mts_data->device = device;
                found_requested_device = true;
                break;
            } else if (device.is_mps() && strcmp(requested_device, "mps") == 0) {
                mts_data->device = device;
                found_requested_device = true;
                break;
            }
        }

        if (!found_requested_device) {
            error->all(FLERR,
                "failed to find requested device ({}): it is either "
                "not supported by this model or not available on this machine",
                requested_device
            );
        }
    }

    mts_data->model->to(mts_data->device);

    // Handle potential mismatch between Kokkos and model devices
    if (std::is_same<LMPDeviceType, Kokkos::Cuda>::value) {
        if (!mts_data->device.is_cuda()) {
            throw std::runtime_error("Kokkos is running on a GPU, but the model is not on a GPU");
        }
    } else {
        if (!mts_data->device.is_cpu()) {
            throw std::runtime_error("Kokkos is running on the host, but the model is not on CPU");
        }
    }

    auto message = "Running simulation on " + mts_data->device.str() + " device with " + mts_data->capabilities->dtype() + " data";
    if (screen) {
        fprintf(screen, "%s\n", message.c_str());
    }
    if (logfile) {
        fprintf(logfile,"%s\n", message.c_str());
    }

    if (!allocated) {
        allocate();
    }

    // this will allow us to receive the NL in a GPU-friendly format
    this->lmp->kokkos->neigh_transpose = 1;

    std::cout << "Running on " << typeid(ExecutionSpaceFromDevice<LMPDeviceType>::space).name() << std::endl;
}


template<class LMPDeviceType>
void PairMetatensorKokkos<LMPDeviceType>::allocate() {
    allocated = 1;

    // setflags stores whether the coeff for a given pair of atom types are known
    /// I'm tempted to change this one to kokkos but I can't find how it's used
    /// Commented out for now
    setflag = memory->create(
        setflag,
        atom->ntypes + 1,
        atom->ntypes + 1,
        "pair:setflag"
    );

    for (int i = 1; i <= atom->ntypes; i++) {
        for (int j = i; j <= atom->ntypes; j++) {
            setflag[i][j] = 0;
        }
    }

    /// I noticed that this cutsq isn't used in the code and is not
    /// necessary to run it. Commented out for now

    // cutsq stores the squared cutoff for each pair
    cutsq = memory->create(
        cutsq,
        atom->ntypes + 1,
        atom->ntypes + 1,
        "pair:cutsq"
    );

    // lammps_types_to_species stores the mapping from lammps atom types to
    // the metatensor model species
    /// This will stay non-kokkos for now (only used at initialization)
    type_mapping = memory->create(
        type_mapping,
        atom->ntypes + 1,
        "PairMetatensor:type_mapping"
    );

    for (int i = 1; i <= atom->ntypes; i++) {
        type_mapping[i] = -1;
    }
}

template<class LMPDeviceType>
double PairMetatensorKokkos<LMPDeviceType>::init_one(int, int) {
    return mts_data->max_cutoff;
}


// called on pair_coeff
template<class LMPDeviceType>
void PairMetatensorKokkos<LMPDeviceType>::coeff(int argc, char ** argv) {
    if (argc < 3 || strcmp(argv[0], "*") != 0 || strcmp(argv[1], "*") != 0) {
        error->all(FLERR, "invalid pair_coeff, expected `pair_coeff * * <list of types>`");
    }

    if (atom->ntypes != argc - 2) {
        error->all(FLERR,
            "invalid pair_coeff, expected `pair_coeff * * <list of types>` with {} types",
            atom->ntypes
        );
    }

    for (int lammps_type=1; lammps_type<argc - 1; lammps_type++) {
        int type = utils::inumeric(FLERR, argv[lammps_type + 1], true, lmp);
        type_mapping[lammps_type] = type;
    }

    // mark all pairs coeffs as known
    for (int i = 1; i <= atom->ntypes; i++) {
        for (int j = 1; j <= atom->ntypes; j++) {
            setflag[i][j] = 1;
            setflag[j][i] = 1;
        }
    }
}


// called when the run starts
template<class LMPDeviceType>
void PairMetatensorKokkos<LMPDeviceType>::init_style() {
    // Require newton pair on since we need to communicate forces accumulated on
    // ghost atoms to neighboring domains. These forces contributions come from
    // gradient of a local descriptor w.r.t. domain ghosts (periodic images
    // ghosts are handled separately).
    /// Would be good if we could change this because Newton off is the Kokkos default
    if (force->newton_pair != 1) {
        error->all(FLERR, "Pair style metatensor requires newton pair on");
    }

    // get the model's interaction range
    auto range = mts_data->capabilities->engine_interaction_range(mts_data->evaluation_options->length_unit());
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
        auto requested_nl = mts_data->model->run_method("requested_neighbor_lists");
        for (const auto& ivalue: requested_nl.toList()) {
            auto options = ivalue.get().toCustomClass<metatensor_torch::NeighborListOptionsHolder>();
            auto cutoff = options->engine_cutoff(mts_data->evaluation_options->length_unit());

            mts_data->max_cutoff = std::max(mts_data->max_cutoff, cutoff);
        }
    } else {
        mts_data->max_cutoff = range;
    }

    if (!std::isfinite(mts_data->max_cutoff)) {
        error->all(FLERR,
            "the largest cutoff of this model is infinite, "
            "we can't compute the corresponding neighbor list"
        );
    }

    /// create Kokkos view for type_mapping
    Kokkos::View<int32_t*, Kokkos::LayoutRight, LMPDeviceType> type_mapping_kokkos("type_mapping", atom->ntypes + 1);
    /// copy type_mapping to the Kokkos view (via a host mirror view)
    auto type_mapping_kokkos_host = Kokkos::create_mirror_view(type_mapping_kokkos);
    for (int i = 0; i < atom->ntypes + 1; i++) {
        type_mapping_kokkos_host(i) = type_mapping[i];
    }
    Kokkos::deep_copy(type_mapping_kokkos, type_mapping_kokkos_host);

    // create system adaptor
    auto options = MetatensorSystemOptionsKokkos<LMPDeviceType>{
        this->type_mapping,
        type_mapping_kokkos,
        mts_data->max_cutoff,
        mts_data->check_consistency,
    };
    mts_data->system_adaptor = std::make_unique<MetatensorSystemAdaptorKokkos<LMPDeviceType>>(lmp, this, options);

    // Translate from the metatensor neighbor lists requests to LAMMPS neighbor
    // lists requests.
    auto requested_nl = mts_data->model->run_method("requested_neighbor_lists");
    for (const auto& ivalue: requested_nl.toList()) {
        auto options = ivalue.get().toCustomClass<metatensor_torch::NeighborListOptionsHolder>();
        auto cutoff = options->engine_cutoff(mts_data->evaluation_options->length_unit());
        assert(cutoff <= mts_data->max_cutoff);

        mts_data->system_adaptor->add_nl_request(cutoff, options);
    }
}


template<class LMPDeviceType>
void PairMetatensorKokkos<LMPDeviceType>::init_list(int id, NeighList *ptr) {
    mts_data->system_adaptor->init_list(id, ptr);
}


template<class LMPDeviceType>
void PairMetatensorKokkos<LMPDeviceType>::compute(int eflag, int vflag) {
    // auto start = std::chrono::high_resolution_clock::now();
    // auto end = std::chrono::high_resolution_clock::now();

    // auto x = atomKK->k_x.view<LMPDeviceType>();
    // auto h_array = Kokkos::create_mirror_view(d_array);
    // Kokkos::deep_copy(h_array, d_array);
    // // Print the values on the host
    // for (int i = 0; i < 32; ++i) {
    //     for (int j = 0; j < 3; ++j) {
    //         std::cout << h_array(i, j) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    /// Declare what we need to read from the atomKK object and what we will modify
    this->atomKK->sync(ExecutionSpaceFromDevice<LMPDeviceType>::space, X_MASK | F_MASK | TAG_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK);
    this->atomKK->modified(ExecutionSpaceFromDevice<LMPDeviceType>::space, ENERGY_MASK | F_MASK | VIRIAL_MASK);

    if (eflag || vflag) {
        ev_setup(eflag, vflag);
    } else {
        evflag = vflag_fdotr = eflag_global = eflag_atom = 0;
    }

    if (eflag_atom) {
        mts_data->evaluation_options->outputs.at("energy")->per_atom = true;
    } else {
        mts_data->evaluation_options->outputs.at("energy")->per_atom = false;
    }

    auto dtype = torch::kFloat64;
    if (mts_data->capabilities->dtype() == "float64") {
        dtype = torch::kFloat64;
    } else if (mts_data->capabilities->dtype() == "float32") {
        dtype = torch::kFloat32;
    } else {
        error->all(FLERR, "the model requested an unsupported dtype '{}'", mts_data->capabilities->dtype());
    }

    // torch::cuda::synchronize();
    // start = std::chrono::high_resolution_clock::now();

    // transform from LAMMPS to metatensor System
    auto system = mts_data->system_adaptor->system_from_lmp(
        static_cast<bool>(vflag_global), mts_data->remap_pairs, dtype, mts_data->device
    );

    // torch::cuda::synchronize();
    // end = std::chrono::high_resolution_clock::now();
    // std::cout << "sys-from-lmp: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << " ms" << std::endl;

    // only run the calculation for atoms actually in the current domain
    auto tensor_options = torch::TensorOptions().dtype(torch::kInt32).device(mts_data->device);
    torch::Tensor selected_atoms_values = torch::stack({
        torch::zeros({atom->nlocal}, tensor_options),
        torch::arange(atom->nlocal, tensor_options)
    }, -1);

    auto selected_atoms = torch::make_intrusive<metatensor_torch::LabelsHolder>(
        std::vector<std::string>{"system", "atom"}, selected_atoms_values
    );
    mts_data->evaluation_options->set_selected_atoms(selected_atoms);

    torch::IValue result_ivalue;

    // torch::cuda::synchronize();
    // start = std::chrono::high_resolution_clock::now();

    try {
        result_ivalue = mts_data->model->forward({
            std::vector<metatensor_torch::System>{system},
            mts_data->evaluation_options,
            mts_data->check_consistency
        });
    } catch (const std::exception& e) {
        error->all(FLERR, "error evaluating the torch model: {}", e.what());
    }

    // torch::cuda::synchronize();
    // end = std::chrono::high_resolution_clock::now();
    // std::cout << "Time taken forward: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << " ms" << std::endl;

    auto result = result_ivalue.toGenericDict();
    auto energy = result.at("energy").toCustomClass<metatensor_torch::TensorMapHolder>();
    auto energy_tensor = metatensor_torch::TensorMapHolder::block_by_id(energy, 0)->values();
    auto energy_detached = energy_tensor.detach().to(torch::kCPU).to(torch::kFloat64);

    // store the energy returned by the model
    torch::Tensor global_energy;
    if (eflag_atom) {
        auto energies = energy_detached.accessor<double, 2>();
        for (int i=0; i<atom->nlocal + atom->nghost; i++) {
            // TODO: handle out of order samples
            eatom[i] += energies[i][0];
        }

        global_energy = energy_detached.sum(0);
        assert(energy_detached.sizes() == std::vector<int64_t>({1}));
    } else {
        assert(energy_detached.sizes() == std::vector<int64_t>({1, 1}));
        global_energy = energy_detached.reshape({1});
    }

    if (eflag_global) {
        eng_vdwl += global_energy.item<double>();
    }

    // reset gradients to zero before calling backward
    mts_data->system_adaptor->positions.mutable_grad() = torch::Tensor();
    mts_data->system_adaptor->strain.mutable_grad() = torch::Tensor();

    // compute forces/virial with backward propagation

    // torch::cuda::synchronize();
    // start = std::chrono::high_resolution_clock::now();

    energy_tensor.backward(-torch::ones_like(energy_tensor));

    // torch::cuda::synchronize();
    // end = std::chrono::high_resolution_clock::now();
    // std::cout << "Time taken backward: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << " ms" << std::endl;

    auto forces_tensor = mts_data->system_adaptor->positions.grad();
    assert(forces_tensor.scalar_type() == torch::kFloat64);

    auto forces_lammps_kokkos = this->atomKK->k_f. template view<LMPDeviceType>();
    /// Is it possible to do double*[3] here?
    auto forces_metatensor_kokkos = Kokkos::View<double**, Kokkos::LayoutRight, LMPDeviceType, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(forces_tensor.contiguous().data_ptr<double>(), atom->nlocal + atom->nghost, 3);

    Kokkos::parallel_for("PairMetatensorKokkos::compute::force_accumulation", atom->nlocal + atom->nghost, KOKKOS_LAMBDA(const int i) {
        forces_lammps_kokkos(i, 0) += forces_metatensor_kokkos(i, 0);
        forces_lammps_kokkos(i, 1) += forces_metatensor_kokkos(i, 1);
        forces_lammps_kokkos(i, 2) += forces_metatensor_kokkos(i, 2);
    });

    assert(!vflag_fdotr);

    if (vflag_global) {
        auto virial_tensor = mts_data->system_adaptor->strain.grad();
        assert(virial_tensor.scalar_type() == torch::kFloat64);

        // apparently the cell is not supported in Kokkos format,
        // so it has to be updated on CPU (??)
        auto predicted_virial_tensor_cpu = virial_tensor.cpu();
        auto predicted_virial = predicted_virial_tensor_cpu.accessor<double, 2>();

        virial[0] += predicted_virial[0][0];
        virial[1] += predicted_virial[1][1];
        virial[2] += predicted_virial[2][2];

        virial[3] += 0.5 * (predicted_virial[1][0] + predicted_virial[0][1]);
        virial[4] += 0.5 * (predicted_virial[2][0] + predicted_virial[0][2]);
        virial[5] += 0.5 * (predicted_virial[2][1] + predicted_virial[1][2]);
    }

    if (vflag_atom) {
        error->all(FLERR, "per atom virial is not implemented");
    }
}

namespace LAMMPS_NS {
template class PairMetatensorKokkos<LMPDeviceType>;
/// TODO: Host version
}
