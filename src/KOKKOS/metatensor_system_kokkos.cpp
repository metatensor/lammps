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
#include "metatensor_system_kokkos.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "neighbor.h"

#include "neigh_list.h"
#include "neigh_request.h"

#include "kokkos.h"
#include "atom_kokkos.h"

#include <torch/cuda.h>
#include <chrono>

#ifndef KOKKOS_ENABLE_CUDA
namespace Kokkos {
class Cuda {};
} // namespace Kokkos
#endif // KOKKOS_ENABLE_CUDA

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class LMPDeviceType>
MetatensorSystemAdaptorKokkos<LMPDeviceType>::MetatensorSystemAdaptorKokkos(LAMMPS *lmp, Pair* requestor, MetatensorSystemOptionsKokkos<LMPDeviceType> options):
    Pointers(lmp),
    list_(nullptr),
    options_(std::move(options)),
    caches_(),
    atomic_types_(torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt32)))
{
    torch::Device device = torch::kCPU;
    if (std::is_same<LMPDeviceType, Kokkos::Cuda>::value) {
        device = torch::kCUDA;
    } else {
        device = torch::kCPU;
    }

    // We ask LAMMPS for a full neighbor lists because we need to know about
    // ALL pairs, even if options->full_list() is false. We will then filter
    // the pairs to only include each pair once where needed.
    auto request = neighbor->add_request(requestor, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
    request->set_id(0);
    request->set_cutoff(options_.interaction_range);
    request->set_kokkos_host(0);
    request->set_kokkos_device(1);

    this->strain = torch::eye(3, torch::TensorOptions().dtype(torch::kFloat64).device(device).requires_grad(true));
}

template<class LMPDeviceType>
MetatensorSystemAdaptorKokkos<LMPDeviceType>::MetatensorSystemAdaptorKokkos(LAMMPS *lmp, Compute* requestor, MetatensorSystemOptionsKokkos<LMPDeviceType> options):
    Pointers(lmp),
    list_(nullptr),
    options_(std::move(options)),
    caches_(),
    atomic_types_(torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt32)))
{
    torch::Device device = torch::kCPU;
    if (std::is_same<LMPDeviceType, Kokkos::Cuda>::value) {
        device = torch::kCUDA;
    } else {
        device = torch::kCPU;
    }

    auto request = neighbor->add_request(requestor, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
    request->set_id(0);
    request->set_cutoff(options_.interaction_range);

    this->strain = torch::eye(3, torch::TensorOptions().dtype(torch::kFloat64).device(device).requires_grad(true));
}

template<class LMPDeviceType>
MetatensorSystemAdaptorKokkos<LMPDeviceType>::~MetatensorSystemAdaptorKokkos() {

}

template<class LMPDeviceType>
void MetatensorSystemAdaptorKokkos<LMPDeviceType>::init_list(int id, NeighList* ptr) {
    assert(id == 0);
    list_ = ptr;
}

template<class LMPDeviceType>
void MetatensorSystemAdaptorKokkos<LMPDeviceType>::add_nl_request(double cutoff, metatensor_torch::NeighborListOptions request) {
    if (cutoff > options_.interaction_range) {
        error->all(FLERR,
            "Invalid metatensor model: one of the requested neighbor lists "
            "has a cutoff ({}) larger than the model interaction range ({})",
            cutoff, options_.interaction_range
        );
    } else if (cutoff < 0 || !std::isfinite(cutoff)) {
        error->all(FLERR,
            "model requested an invalid cutoff for neighbors list: {} "
            "(cutoff in model units is {})",
            cutoff, request->cutoff()
        );
    }

    caches_.push_back({
        cutoff,
        request,
        /*known_samples = */ {},
        /*samples = */ {},
        /*distances_f64 = */ {},
        /*distances_f32 = */ {},
    });
}


template<class LMPDeviceType>
void MetatensorSystemAdaptorKokkos<LMPDeviceType>::setup_neighbors_remap(metatensor_torch::System& system) {
    // auto start = std::chrono::high_resolution_clock::now();
    // auto end = std::chrono::high_resolution_clock::now();

    auto dtype = system->positions().scalar_type();
    auto device = system->positions().device();

    auto positions_kokkos = this->atomKK->k_x. template view<LMPDeviceType>();
    auto total_n_atoms = atomKK->nlocal + atomKK->nghost;

    // torch::cuda::synchronize();
    // start = std::chrono::high_resolution_clock::now();
    
    /*-------------- whatever, this will be done on CPU for now ------------------------*/
    // The cost of this section seems to be very low

    // There is no kokkos cell in LAMMPS, so we need to transfer
    auto cell_inv_tensor = system->cell().inverse().t().to(device).to(dtype);

    // Collect the local atom id of all local & ghosts atoms, mapping ghosts
    // atoms which are periodic images of local atoms back to the local atoms.
    //
    // Metatensor expects pairs corresponding to periodic atoms to be between
    // the main atoms, but using the actual distance vector between the atom and
    // the ghost.
    original_atom_id_.clear();
    original_atom_id_.reserve(total_n_atoms);

    // identify all local atom by their LAMMPS atom tag.
    local_atoms_tags_.clear();
    for (int i=0; i<atom->nlocal; i++) {
        original_atom_id_.emplace_back(i);
        local_atoms_tags_.emplace(atom->tag[i], i);
    }

    // now loop over ghosts & map them back to the main cell if needed
    ghost_atoms_tags_.clear();
    for (int i=atom->nlocal; i<total_n_atoms; i++) {
        auto tag = atom->tag[i];
        auto it = local_atoms_tags_.find(tag);
        if (it != local_atoms_tags_.end()) {
            // this is the periodic image of an atom already owned by this domain
            original_atom_id_.emplace_back(it->second);
        } else {
            // this can either be a periodic image of an atom owned by another
            // domain, or directly an atom from another domain. Since we can not
            // really distinguish between these, we take the first atom as the
            // "main" one and remap all atoms with the same tag to the first one
            auto it = ghost_atoms_tags_.find(tag);
            if (it != ghost_atoms_tags_.end()) {
                // we already found this atom elsewhere in the system
                original_atom_id_.emplace_back(it->second);
            } else {
                // this is the first time we are seeing this atom
                original_atom_id_.emplace_back(i);
                ghost_atoms_tags_.emplace(tag, i);
            }
        }
    }

    auto original_atom_id_tensor = torch::from_blob(
        original_atom_id_.data(),
        {total_n_atoms},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
    );
    original_atom_id_tensor = original_atom_id_tensor.to(device);  // RIP

    // torch::cuda::synchronize();
    // end = std::chrono::high_resolution_clock::now();
    // std::cout << "  ghost mapping (CPU): " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms" << std::endl;

    /*----------- end of whatever, this will be done on CPU for now --------------*/

    // torch::cuda::synchronize();
    // start = std::chrono::high_resolution_clock::now();

    NeighListKokkos<LMPDeviceType>* list_kk = static_cast<NeighListKokkos<LMPDeviceType>*>(this->list_);

    auto numneigh_kk = list_kk->d_numneigh;
    auto neighbors_kk = list_kk->d_neighbors_transpose;
    auto ilist_kk = list_kk->d_ilist;

    auto max_number_of_neighbors = list_kk->maxneighs;

    // mask neighbors_kk with NEIGHMASK
    Kokkos::View<int**, Kokkos::LayoutRight, LMPDeviceType> neighbors_kk_masked("neighbors_kk_masked", total_n_atoms, max_number_of_neighbors);
    Kokkos::parallel_for("mask_neigh", total_n_atoms*max_number_of_neighbors, KOKKOS_LAMBDA(int i) {
        auto local_i = i / max_number_of_neighbors;
        auto local_j = i % max_number_of_neighbors;
        neighbors_kk_masked(local_i, local_j) = neighbors_kk(local_i, local_j) & NEIGHMASK;
    });

    auto numneigh_torch = torch::from_blob(
        numneigh_kk.data(),
        {total_n_atoms},
        torch::TensorOptions().dtype(torch::kInt32).device(device)
    );

    auto neighbors_torch = torch::from_blob(
        neighbors_kk_masked.data(),
        {total_n_atoms, max_number_of_neighbors},
        torch::TensorOptions().dtype(torch::kInt32).device(device)
    );

    auto ilist_torch = torch::from_blob(
        ilist_kk.data(),
        {total_n_atoms},
        torch::TensorOptions().dtype(torch::kInt32).device(device)
    );

    auto expanded_arange = torch::arange(max_number_of_neighbors, torch::TensorOptions().dtype(torch::kInt32).device(device)).unsqueeze(0).expand({total_n_atoms, -1});
    auto neighbor_2d_mask = expanded_arange < numneigh_torch.unsqueeze(1);

    auto expanded_arange_other_dim = torch::arange(total_n_atoms, torch::TensorOptions().dtype(torch::kInt32).device(device)).unsqueeze(1).expand({-1, max_number_of_neighbors});
    auto index_for_ilist = expanded_arange_other_dim.masked_select(neighbor_2d_mask);
    auto centers_tensor = ilist_torch.index_select(0, index_for_ilist);

    auto neighbors_tensor = neighbors_torch.masked_select(neighbor_2d_mask);

    // torch::cuda::synchronize();
    // end = std::chrono::high_resolution_clock::now();
    // std::cout << "  NL format conversion: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms" << std::endl;

    // torch::cuda::synchronize();
    // start = std::chrono::high_resolution_clock::now();

    // change centers and neighbors to the original atom ids
    auto centers_tensor_original_id = original_atom_id_tensor.index_select(0, centers_tensor);
    auto neighbors_tensor_original_id = original_atom_id_tensor.index_select(0, neighbors_tensor);

    // create torch tensor with the positions (TEMPORARY, TODO: change)
    auto positions_tensor = torch::from_blob(
        positions_kokkos.data(),
        {total_n_atoms, 3},
        torch::TensorOptions().dtype(torch::kFloat64).device(device)
    ).to(dtype);

    for (auto& cache: caches_) {
        // half list mask, if necessary (TODO: change names! This could modify the tensors outside the loop if more than one NL!)
        auto full_list = cache.options->full_list();
        if (!full_list) {
            auto half_list_mask = centers_tensor_original_id <= neighbors_tensor_original_id;
            centers_tensor = centers_tensor.masked_select(half_list_mask);
            neighbors_tensor = neighbors_tensor.masked_select(half_list_mask);
            centers_tensor_original_id = centers_tensor_original_id.masked_select(half_list_mask);
            neighbors_tensor_original_id = neighbors_tensor_original_id.masked_select(half_list_mask);
        }

        // distance mask
        auto interatomic_vectors = positions_tensor.index_select(0, neighbors_tensor) - positions_tensor.index_select(0, centers_tensor);
        auto distance_mask = torch::sum(interatomic_vectors.pow(2), 1) < cache.cutoff*cache.cutoff;

        // index everything with the mask
        auto centers_tensor_original_id_filtered = centers_tensor_original_id.masked_select(distance_mask);
        auto neighbors_tensor_original_id_filtered = neighbors_tensor_original_id.masked_select(distance_mask);
        auto interatomic_vectors_filtered = interatomic_vectors.index({distance_mask, torch::indexing::Slice()});

        // find filtered interatomic vectors using the original atoms
        auto interatomic_vectors_original_filtered = positions_tensor.index_select(0, neighbors_tensor_original_id_filtered) - positions_tensor.index_select(0, centers_tensor_original_id_filtered);

        // cell shifts
        auto pair_shifts = interatomic_vectors_filtered - interatomic_vectors_original_filtered;
        auto cell_shifts = pair_shifts.matmul(cell_inv_tensor);
        cell_shifts = torch::round(cell_shifts).to(torch::kInt32);

        if (!full_list) {
            auto half_list_cell_mask = centers_tensor_original_id_filtered == neighbors_tensor_original_id_filtered;
            auto negative_half_space_mask = torch::sum(cell_shifts, 1) < 0;
            // reproduce this mask with torch:
            // if ((shift[0] + shift[1] + shift[2] == 0) && (shift[2] < 0 || (shift[2] == 0 && shift[1] < 0)))
            auto edge_mask = (
                torch::sum(cell_shifts, 1) == 0 & (
                    cell_shifts.index({torch::indexing::Slice(), 2}) < 0 | (
                        cell_shifts.index({torch::indexing::Slice(), 2}) == 0 &
                        cell_shifts.index({torch::indexing::Slice(), 1}) < 0
                    )
                )
            );
            auto final_mask = torch::logical_not(half_list_cell_mask & (negative_half_space_mask | edge_mask));
            centers_tensor_original_id_filtered = centers_tensor_original_id_filtered.masked_select(final_mask);
            neighbors_tensor_original_id_filtered = neighbors_tensor_original_id_filtered.masked_select(final_mask);
            interatomic_vectors_filtered = interatomic_vectors_filtered.index({final_mask, torch::indexing::Slice()});
            cell_shifts = cell_shifts.index({final_mask, torch::indexing::Slice()});
        }

        centers_tensor_original_id_filtered = centers_tensor_original_id_filtered.unsqueeze(-1);
        neighbors_tensor_original_id_filtered = neighbors_tensor_original_id_filtered.unsqueeze(-1);
        auto samples_values = torch::concatenate({centers_tensor_original_id_filtered, neighbors_tensor_original_id_filtered, cell_shifts}, 1);

        auto [samples_values_unique, samples_inverse, _] = torch::unique_dim(
            samples_values, /*dim=*/0, /*sorted=*/true, /*return_inverse=*/true, /*return_counts=*/false
        );

        auto permutation = torch::arange(samples_inverse.size(0), samples_inverse.options());
        samples_inverse = samples_inverse.flip({0});
        permutation = permutation.flip({0});

        auto sample_indices = torch::empty(samples_values_unique.size(0), samples_inverse.options());
        sample_indices.scatter_(0, samples_inverse, permutation);

        // torch::cuda::synchronize();
        // end = std::chrono::high_resolution_clock::now();
        // std::cout << "  NL filtering: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms" << std::endl;

        // torch::cuda::synchronize();
        // start = std::chrono::high_resolution_clock::now();

        auto samples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            std::vector<std::string>{"first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"},
            samples_values_unique
        );

        // torch::cuda::synchronize();
        // end = std::chrono::high_resolution_clock::now();
        // std::cout << "  Time to create big labels: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 << "ms" << std::endl;

        auto neighbor_list = torch::make_intrusive<metatensor_torch::TensorBlockHolder>(
            interatomic_vectors_filtered.index_select(0, sample_indices).unsqueeze(-1),
            samples->to(device),
            std::vector<metatensor_torch::TorchLabels>{
                metatensor_torch::LabelsHolder::create({"xyz"}, {{0}, {1}, {2}})->to(device),
            },
            metatensor_torch::LabelsHolder::create({"distance"}, {{0}})->to(device)
        );

        metatensor_torch::register_autograd_neighbors(system, neighbor_list, options_.check_consistency);
        system->add_neighbor_list(cache.options, neighbor_list);
    }

}


template<class LMPDeviceType>
void MetatensorSystemAdaptorKokkos<LMPDeviceType>::setup_neighbors_no_remap(metatensor_torch::System& system) {
    auto dtype = system->positions().scalar_type();
    auto device = system->positions().device();

    auto positions_kokkos = this->atomKK->k_x. template view<LMPDeviceType>();
    auto total_n_atoms = atomKK->nlocal + atomKK->nghost;

    auto cell_inv_tensor = system->cell().inverse().t().to(device).to(dtype);
    // it might be a good idea to have this as float32 if the model is using float32
    // to speed up the computation, especially on GPU


    /*-------------- whatever, this will be done on CPU for now ------------------------*/

    // Collect the local atom id of all local & ghosts atoms, mapping ghosts
    // atoms which are periodic images of local atoms back to the local atoms.
    //
    // Metatensor expects pairs corresponding to periodic atoms to be between
    // the main atoms, but using the actual distance vector between the atom and
    // the ghost.
    original_atom_id_.clear();
    original_atom_id_.reserve(total_n_atoms);

    // identify all local atom by their LAMMPS atom tag.
    local_atoms_tags_.clear();
    for (int i=0; i<atom->nlocal; i++) {
        original_atom_id_.emplace_back(i);
        local_atoms_tags_.emplace(atom->tag[i], i);
    }

    // now loop over ghosts & map them back to the main cell if needed
    ghost_atoms_tags_.clear();
    for (int i=atom->nlocal; i<total_n_atoms; i++) {
        auto tag = atom->tag[i];
        auto it = local_atoms_tags_.find(tag);
        if (it != local_atoms_tags_.end()) {
            // this is the periodic image of an atom already owned by this domain
            original_atom_id_.emplace_back(it->second);
        } else {
            // this can either be a periodic image of an atom owned by another
            // domain, or directly an atom from another domain. Since we can not
            // really distinguish between these, we take the first atom as the
            // "main" one and remap all atoms with the same tag to the first one
            auto it = ghost_atoms_tags_.find(tag);
            if (it != ghost_atoms_tags_.end()) {
                // we already found this atom elsewhere in the system
                original_atom_id_.emplace_back(it->second);
            } else {
                // this is the first time we are seeing this atom
                original_atom_id_.emplace_back(i);
                ghost_atoms_tags_.emplace(tag, i);
            }
        }
    }
    /*----------- end of whatever, this will be done on CPU for now --------------*/

    auto original_atom_id_tensor = torch::from_blob(
        original_atom_id_.data(),
        {total_n_atoms},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
    );
    original_atom_id_tensor = original_atom_id_tensor.to(device);  // RIP

    // Accumulate total number of pairs
    int total_number_of_pairs = 0;
    for (int ii=0; ii<(list_->inum + list_->gnum); ii++) {
        total_number_of_pairs += list_->numneigh[ii];
    }
    std::vector<int> centers(total_number_of_pairs);
    std::vector<int> neighbors(total_number_of_pairs);

    // Fill the centers and neighbors arrays with the original atom ids
    int pair_index = 0;
    for (int ii=0; ii<(list_->inum + list_->gnum); ii++) {
        auto atom_i = list_->ilist[ii];
        auto neighbors_ii = list_->firstneigh[ii];
        for (int jj=0; jj<list_->numneigh[ii]; jj++) {
            centers[pair_index] = atom_i;
            neighbors[pair_index] = neighbors_ii[jj] & NEIGHMASK;
            pair_index++;
        }
    }

    // Create torch tensors for the centers and neighbors arrays
    auto centers_tensor = torch::from_blob(
        centers.data(),
        {total_number_of_pairs},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
    );
    centers_tensor = centers_tensor.to(device);
    auto neighbors_tensor = torch::from_blob(
        neighbors.data(),
        {total_number_of_pairs},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU)
    );
    neighbors_tensor = neighbors_tensor.to(device);

    // change centers and neighbors to the original atom ids
    auto centers_tensor_original_id = original_atom_id_tensor.index_select(0, centers_tensor);
    auto neighbors_tensor_original_id = original_atom_id_tensor.index_select(0, neighbors_tensor);

    // create torch tensor with the positions (TEMPORARY, TODO: change)
    auto positions_tensor = torch::from_blob(
        positions_kokkos.data(),
        {total_n_atoms, 3},
        torch::TensorOptions().dtype(torch::kFloat64).device(device)
    ).to(dtype);

    for (auto& cache: caches_) {
        // half list mask, if necessary (TODO: change names! This could modify the tensors outside the loop if more than one NL!)
        auto full_list = cache.options->full_list();
        if (!full_list) {
            auto half_list_mask = centers_tensor_original_id <= neighbors_tensor_original_id;
            centers_tensor = centers_tensor.masked_select(half_list_mask);
            neighbors_tensor = neighbors_tensor.masked_select(half_list_mask);
            centers_tensor_original_id = centers_tensor_original_id.masked_select(half_list_mask);
            neighbors_tensor_original_id = neighbors_tensor_original_id.masked_select(half_list_mask);
        }

        // distance mask
        auto interatomic_vectors = positions_tensor.index_select(0, neighbors_tensor) - positions_tensor.index_select(0, centers_tensor);
        auto distance_mask = torch::sum(interatomic_vectors.pow(2), 1) < cache.cutoff*cache.cutoff;

        // index everything with the mask
        auto centers_tensor_original_id_filtered = centers_tensor_original_id.masked_select(distance_mask);
        auto neighbors_tensor_original_id_filtered = neighbors_tensor_original_id.masked_select(distance_mask);
        auto interatomic_vectors_filtered = interatomic_vectors.index({distance_mask, torch::indexing::Slice()});

        // find filtered interatomic vectors using the original atoms
        auto interatomic_vectors_original_filtered = positions_tensor.index_select(0, neighbors_tensor_original_id_filtered) - positions_tensor.index_select(0, centers_tensor_original_id_filtered);

        // cell shifts
        auto pair_shifts = interatomic_vectors_filtered - interatomic_vectors_original_filtered;
        auto cell_shifts = pair_shifts.matmul(cell_inv_tensor);
        cell_shifts = torch::round(cell_shifts).to(torch::kInt32);

        if (!full_list) {
            auto half_list_cell_mask = centers_tensor_original_id_filtered == neighbors_tensor_original_id_filtered;
            auto negative_half_space_mask = torch::sum(cell_shifts, 1) < 0;
            // reproduce this mask with torch:
            // if ((shift[0] + shift[1] + shift[2] == 0) && (shift[2] < 0 || (shift[2] == 0 && shift[1] < 0)))
            auto edge_mask = (
                torch::sum(cell_shifts, 1) == 0 & (
                    cell_shifts.index({torch::indexing::Slice(), 2}) < 0 | (
                        cell_shifts.index({torch::indexing::Slice(), 2}) == 0 &
                        cell_shifts.index({torch::indexing::Slice(), 1}) < 0
                    )
                )
            );
            auto final_mask = torch::logical_not(half_list_cell_mask & (negative_half_space_mask | edge_mask));
            centers_tensor_original_id_filtered = centers_tensor_original_id_filtered.masked_select(final_mask);
            neighbors_tensor_original_id_filtered = neighbors_tensor_original_id_filtered.masked_select(final_mask);
            interatomic_vectors_filtered = interatomic_vectors_filtered.index({final_mask, torch::indexing::Slice()});
            cell_shifts = cell_shifts.index({final_mask, torch::indexing::Slice()});
        }

        centers_tensor_original_id_filtered = centers_tensor_original_id_filtered.unsqueeze(-1);
        neighbors_tensor_original_id_filtered = neighbors_tensor_original_id_filtered.unsqueeze(-1);
        auto samples_values = torch::concatenate({centers_tensor_original_id_filtered, neighbors_tensor_original_id_filtered, cell_shifts}, 1);

        auto [samples_values_unique, samples_inverse, _] = torch::unique_dim(
            samples_values, /*dim=*/0, /*sorted=*/true, /*return_inverse=*/true, /*return_counts=*/false
        );

        auto permutation = torch::arange(samples_inverse.size(0), samples_inverse.options());
        samples_inverse = samples_inverse.flip({0});
        permutation = permutation.flip({0});

        auto sample_indices = torch::empty(samples_values_unique.size(0), samples_inverse.options());
        sample_indices.scatter_(0, samples_inverse, permutation);

        auto samples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
            std::vector<std::string>{"first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"},
            samples_values_unique
        );

        auto neighbor_list = torch::make_intrusive<metatensor_torch::TensorBlockHolder>(
            interatomic_vectors_filtered.index_select(0, sample_indices).unsqueeze(-1).to(dtype).to(device),
            samples->to(device),
            std::vector<metatensor_torch::TorchLabels>{
                metatensor_torch::LabelsHolder::create({"xyz"}, {{0}, {1}, {2}})->to(device),
            },
            metatensor_torch::LabelsHolder::create({"distance"}, {{0}})->to(device)
        );

        metatensor_torch::register_autograd_neighbors(system, neighbor_list, options_.check_consistency);
        system->add_neighbor_list(cache.options, neighbor_list);
    }
}


template<class LMPDeviceType>
metatensor_torch::System MetatensorSystemAdaptorKokkos<LMPDeviceType>::system_from_lmp(
    bool do_virial,
    bool remap_pairs,
    torch::ScalarType dtype,
    torch::Device device
) {
    auto total_n_atoms = atomKK->nlocal + atomKK->nghost;

    auto atom_types_lammps_kokkos = atomKK->k_type.view<LMPDeviceType>();
    auto mapping = options_.types_mapping_kokkos;
    Kokkos::View<int32_t*, Kokkos::LayoutRight, LMPDeviceType> atom_types_metatensor_kokkos("atom_types_metatensor", total_n_atoms);   /// Can be a class member? (allocation alert)

    Kokkos::parallel_for(
        "MetatensorSystemAdaptorKokkos::system_from_lmp::atom_types_mapping",
        Kokkos::RangePolicy(0, total_n_atoms),
        KOKKOS_LAMBDA(int i)
    {
        atom_types_metatensor_kokkos(i) = mapping(atom_types_lammps_kokkos(i));
    });

    atomic_types_ = torch::from_blob(
        atom_types_metatensor_kokkos.data(),
        {total_n_atoms},
        torch::TensorOptions().dtype(torch::kInt32).device(device)
    ).clone();  /// Again, allocation alert. Not sure if this can be avoided

    // atom->x contains "real" and then ghost atoms, in that order
    auto positions_kokkos = atomKK->k_x.view<LMPDeviceType>();
    auto tensor_options_positions = torch::TensorOptions().dtype(torch::kFloat64).device(device);
    this->positions = torch::from_blob(
        positions_kokkos.data(), {total_n_atoms, 3},
        // requires_grad=true since we always need gradients w.r.t. positions
        tensor_options_positions
    ).clone().requires_grad_(true);  /// Allocation alert (clone)

    auto tensor_options_cell = torch::TensorOptions().dtype(dtype).device(device);
    auto cell = torch::zeros({3, 3}, tensor_options_cell);  /// Allocation alert, we could make it a class member and allocate it once
    /// domain doesn't seem to have a Kokkos version
    cell[0][0] = domain->xprd;

    cell[1][0] = domain->xy;
    cell[1][1] = domain->yprd;

    cell[2][0] = domain->xz;
    cell[2][1] = domain->yz;
    cell[2][2] = domain->zprd;
    /// And the other elements? Are they always zero?

    auto system_positions = this->positions.to(dtype);
    cell = cell.to(dtype).to(device);   /// to(device) alert. How do we find the cell on Kokkos?

    if (do_virial) {
        auto model_strain = this->strain.to(dtype);  /// already on the correct device

        // pretend to scale positions/cell by the strain so that
        // it enters the computational graph.
        system_positions = system_positions.matmul(model_strain);
        cell = cell.matmul(model_strain);
    }

    auto system = torch::make_intrusive<metatensor_torch::SystemHolder>(
        atomic_types_,
        system_positions,
        cell
    );

    if (remap_pairs) {
        this->setup_neighbors_remap(system);
    } else {
        this->setup_neighbors_no_remap(system);
    }
    return system;
}

namespace LAMMPS_NS {
template class MetatensorNeighborsDataKokkos<LMPDeviceType>;
template class MetatensorSystemAdaptorKokkos<LMPDeviceType>;
}
