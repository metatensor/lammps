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

/* ----------------------------------------------------------------------
   Fix metatomic/kk: Kokkos version of ML-driven position and momentum prediction

   This is the Kokkos-enabled version of fix metatomic. It uses Kokkos views
   for data access and the MetatomicSystemAdaptorKokkos for efficient data
   transfer between LAMMPS and the ML model.
------------------------------------------------------------------------- */

#include "fix_metatomic_kokkos.h"

#include "error.h"
#include "neigh_request.h"
#include "atom_masks.h"
#include "force.h"
#include "update.h"
#include "neighbor_kokkos.h"

#include "atom_kokkos.h"
#include "metatomic_system_kokkos.h"
#include "metatomic_types.h"

#include <algorithm>
#include <cctype>

using namespace LAMMPS_NS;
using namespace FixConst;

// LAMMPS uses `LAMMPS_NS::tagint` and `int` for tags and neighbor lists, respectively.
// For the moment, we require both to be int32_t for this interface
static_assert(std::is_same_v<LAMMPS_NS::tagint, int32_t>, "Error: LAMMPS_NS::tagint must be int32_t to compile metatomic/kk");
static_assert(std::is_same_v<int, int32_t>, "Error: int must be int32_t to compile metatomic/kk");

template<typename T, class DeviceType>
using UnmanagedView = Kokkos::View<T, Kokkos::LayoutRight, DeviceType, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixMetatomicKokkos<DeviceType>::FixMetatomicKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixMetatomic(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  datamask_read = X_MASK | V_MASK | F_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK;
  datamask_modify = X_MASK | V_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixMetatomicKokkos<DeviceType>::~FixMetatomicKokkos() {}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixMetatomicKokkos<DeviceType>::init()
{
  FixMetatomic::init();

  auto request = neighbor->find_request(this);
  request->set_kokkos_host(
    std::is_same_v<DeviceType, LMPHostType> &&
    !std::is_same_v<DeviceType, LMPDeviceType>
  );
  request->set_kokkos_device(std::is_same_v<DeviceType, LMPDeviceType>);

  // copy type mapping from host to device, to be able to give a device pointer
  // to MetatomicSystemAdaptorKokkos
  auto type_mapping_kk_host = UnmanagedView<int32_t*, LMPHostType>(this->type_mapping, atom->ntypes + 1);
  this->type_mapping_kk = Kokkos::View<int32_t*, Kokkos::LayoutRight, DeviceType>("type_mapping_kk", atom->ntypes + 1);
  Kokkos::deep_copy(this->type_mapping_kk, type_mapping_kk_host);

  auto options = MetatomicSystemOptions{
    this->type_mapping_kk.data(),
    mta_data->max_cutoff,
    mta_data->check_consistency,
    !(mta_data->non_conservative),
  };

  // override the system adaptor with the kokkos version
  this->system_adaptor = std::make_unique<MetatomicSystemAdaptorKokkos<DeviceType>>(lmp, options);

  // request NL with the new adaptor
  auto requested_nl = mta_data->model->run_method("requested_neighbor_lists");
  for (const auto& ivalue: requested_nl.toList()) {
    auto options = ivalue.get().toCustomClass<metatomic_torch::NeighborListOptionsHolder>();
    auto cutoff = options->engine_cutoff(mta_data->evaluation_options->length_unit());
    assert(cutoff <= mta_data->max_cutoff);

    this->system_adaptor->add_nl_request(cutoff, options);
  }

  // Sync mass data to device
  atomKK->k_mass.modify_host();
  atomKK->k_mass.sync<DeviceType>();

  // Allocate Kokkos view for force snapshot
  f_pre_kk = typename AT::t_kkfloat_2d("fix_metatomic:f_pre", atom->nmax, 3);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixMetatomicKokkos<DeviceType>::pick_device(torch::Device* device, const char* requested)
{
  // Pick device based on Kokkos execution space
  *device = KokkosDeviceToTorch<DeviceType>::convert();

  if (requested != nullptr) {
      auto requested_str = std::string(requested);
      std::transform(requested_str.begin(), requested_str.end(), requested_str.begin(), ::tolower);
      if (c10::DeviceTypeName(device->type(), /*lower_case=*/true) != requested_str) {
          error->all(FLERR,
              "requested device '{}' does not match the device being used by kokkos '{}', "
              "use the non-kokkos version of this fix to use a different "
              "device for the model and LAMMPS",
              requested, device->str()
          );
      }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixMetatomicKokkos<DeviceType>::initial_integrate(int /*vflag*/)
{
  // This function performs ML-driven position and momentum updates using Kokkos
  
  // Sync atom data and mark that we will modify positions and velocities
  atomKK->sync(execution_space, datamask_read);
  atomKK->modified(execution_space, datamask_modify);
  
  // Get Kokkos views for atom data
  x = atomKK->k_x.view<DeviceType>();
  v = atomKK->k_v.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();

  int nlocal = atomKK->nlocal;
  int nghost = atomKK->nghost;
  int nall = nlocal + nghost;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  // Determine dtype for the model
  auto dtype = torch::kFloat64;
  if (mta_data->capabilities->dtype() == "float64") {
      dtype = torch::kFloat64;
  } else if (mta_data->capabilities->dtype() == "float32") {
      dtype = torch::kFloat32;
  } else {
      error->all(FLERR, "the model requested an unsupported dtype '{}'", mta_data->capabilities->dtype());
  }

  // Transform from LAMMPS to metatomic System using Kokkos adaptor
  auto system = this->system_adaptor->system_from_lmp(
      mta_list,
      static_cast<bool>(vflag_global),
      mta_data->remap_pairs,
      dtype,
      mta_data->device
  );

  // Gather masses in a tensor and ship to device
  auto float_tensor_options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
  torch::Tensor masses;
  if (rmass.data()) {
      // Per-atom masses: copy from Kokkos view to CPU tensor
      auto rmass_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rmass);
      masses = torch::from_blob(
          rmass_mirror.data(), {nall},
          float_tensor_options.requires_grad(false)
      ).clone().to(mta_data->device);
  } else {
      // Type-based masses: map from atom type to mass
      std::vector<double> masses_vector(nall);
      auto type_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), type);
      auto mass_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mass);
      for (int i = 0; i < nall; i++) {
          masses_vector[i] = mass_mirror[type_mirror[i]];
      }
      masses = torch::from_blob(
          masses_vector.data(), {nall},
          float_tensor_options.requires_grad(false)
      ).clone().to(mta_data->device);
  }
  
  auto label_tensor_options = torch::TensorOptions().dtype(torch::kInt32).device(mta_data->device);
  
  // Add masses to system
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
      masses.to(torch::TensorOptions().dtype(torch::kFloat32)).unsqueeze(-1),
      samples,
      std::vector<metatensor_torch::Labels>{},
      properties
    );
    auto blocks = std::vector<metatensor_torch::TensorBlock>{block};
    auto tmap = torch::make_intrusive<metatensor_torch::TensorMapHolder>(keys, blocks);
    system->add_data("masses", tmap, /*override=*/true);
  }

  // Add momenta to the system
  {
    // Gather velocities from Kokkos view
    auto v_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    auto velocities = torch::from_blob(
        v_mirror.data(), {nall, 3},
        float_tensor_options.requires_grad(false)
    ).clone().to(mta_data->device);

    // Compute momenta = mass * velocity with unit conversion
    // Unit conversion factor for metal units (see fix_metatomic.cpp for details)
    auto momenta = masses.unsqueeze(1) * velocities * (0.001 / 0.09822694743391452);

    // Create TensorBlock for momenta
    auto keys = metatensor_torch::LabelsHolder::single()->to(mta_data->device);
    auto values = momenta.unsqueeze(-1); // add property dimension

    // Define samples
    auto sample_value_components = std::vector<torch::Tensor>{
        torch::zeros(nall, label_tensor_options).unsqueeze(1),
        torch::arange(nall, label_tensor_options).unsqueeze(1)
    };
    auto sample_values = torch::column_stack(sample_value_components);
    metatensor_torch::Labels samples = torch::make_intrusive<metatensor_torch::LabelsHolder>(
        std::vector<std::string>{"system", "atom"}, sample_values
    );

    // Define components
    auto component_values = torch::arange(3, label_tensor_options).unsqueeze(1);
    metatensor_torch::Labels components = torch::make_intrusive<metatensor_torch::LabelsHolder>(
        std::vector<std::string>{"xyz"}, component_values
    );

    auto properties = metatensor_torch::LabelsHolder::single()->to(mta_data->device);
    auto block = torch::make_intrusive<metatensor_torch::TensorBlockHolder>(
      values.to(torch::TensorOptions().dtype(torch::kFloat32)),
      samples,
      std::vector<metatensor_torch::Labels>{components},
      properties
    );
    auto blocks = std::vector<metatensor_torch::TensorBlock>{block};
    auto tmap = torch::make_intrusive<metatensor_torch::TensorMapHolder>(keys, blocks);
    system->add_data("momenta", tmap, /*override=*/true);
  }

  // Configure selected atoms for evaluation
  mta_data->selected_atoms_values.resize_({atomKK->nlocal, 2});
  mta_data->selected_atoms_values.index_put_({torch::indexing::Slice(), 0}, 0);
  auto options = mta_data->selected_atoms_values.options();
  mta_data->selected_atoms_values.index_put_(
      {torch::indexing::Slice(), 1},
      torch::arange(atomKK->nlocal, options)
  );

  auto selected_atoms = torch::make_intrusive<metatensor_torch::LabelsHolder>(
      std::vector<std::string>{"system", "atom"}, mta_data->selected_atoms_values
  );
  mta_data->evaluation_options->set_selected_atoms(selected_atoms);

  // Call the ML model to predict new positions and momenta
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

  // Extract predicted positions (keep on device)
  auto positions_map = result.at("positions").toCustomClass<metatensor_torch::TensorMapHolder>();
  auto positions_block = metatensor_torch::TensorMapHolder::block_by_id(positions_map, 0);
  auto positions = positions_block->values().squeeze(-1).to(mta_data->device).to(torch::kFloat64).contiguous();

  // Extract predicted momenta (keep on device)
  auto momenta_map = result.at("momenta").toCustomClass<metatensor_torch::TensorMapHolder>();
  auto momenta_block = metatensor_torch::TensorMapHolder::block_by_id(momenta_map, 0);
  auto momenta = momenta_block->values().squeeze(-1).to(mta_data->device).to(torch::kFloat64);

  // Convert momenta back from model units to LAMMPS velocity units
  momenta = momenta / (0.001 / 0.09822694743391452);
  momenta = momenta.contiguous();

  // Wrap torch tensors with UnmanagedView for device access
  auto positions_kk = UnmanagedView<double**, DeviceType>(
      positions.template data_ptr<double>(),
      positions.size(0), 3
  );
  auto momenta_kk = UnmanagedView<double**, DeviceType>(
      momenta.template data_ptr<double>(),
      momenta.size(0), 3
  );

  // Get Kokkos views for LAMMPS data
  auto x_view = x;
  auto v_view = v;
  auto mask_view = mask;
  auto type_view = type;
  auto rmass_view = rmass;
  auto mass_view = mass;
  
  // Prepare masses view for device access
  // Copy masses to device if needed
  typename AT::t_kkfloat_1d masses_kk;
  if (rmass.data()) {
      masses_kk = rmass_view;
  } else {
      // Create a per-atom mass array from type-based masses
      masses_kk = typename AT::t_kkfloat_1d("fix_metatomic:masses", nall);
      Kokkos::parallel_for(
          nall,
          KOKKOS_LAMBDA(int i) {
              masses_kk[i] = mass_view[type_view[i]];
          }
      );
  }

  // Apply ML predictions to LAMMPS atoms using Kokkos parallel operations on device
  int groupbit_copy = groupbit;
  Kokkos::parallel_for(
      nlocal,
      KOKKOS_LAMBDA(int i) {
          if (mask_view[i] & groupbit_copy) {
              // Update positions with ML predictions
              x_view(i, 0) = positions_kk(i, 0);
              x_view(i, 1) = positions_kk(i, 1);
              x_view(i, 2) = positions_kk(i, 2);

              // Update velocities from predicted momenta: v = p / m
              double mass_i = masses_kk[i];
              v_view(i, 0) = momenta_kk(i, 0) / mass_i;
              v_view(i, 1) = momenta_kk(i, 1) / mass_i;
              v_view(i, 2) = momenta_kk(i, 2) / mass_i;
          }
      }
  );
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixMetatomicKokkos<DeviceType>::post_force(int /*vflag*/)
{
  // Take a snapshot of forces for Langevin compatibility
  // See fix_metatomic.cpp for detailed explanation
  
  atomKK->sync(execution_space, F_MASK);
  
  auto f_current = atomKK->k_f.view<DeviceType>();
  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  // Resize force snapshot if needed
  if (f_pre_kk.extent(0) < (size_t)atom->nmax) {
      f_pre_kk = typename AT::t_kkfloat_2d("fix_metatomic:f_pre", atom->nmax, 3);
  }

  // Copy current forces to snapshot using Kokkos parallel operations
  auto f_pre_sub = Kokkos::subview(f_pre_kk, std::make_pair(0, nlocal), Kokkos::ALL);
  auto f_current_sub = Kokkos::subview(f_current, std::make_pair(0, nlocal), Kokkos::ALL);
  Kokkos::deep_copy(f_pre_sub, f_current_sub);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixMetatomicKokkos<DeviceType>::final_integrate()
{
  // Apply velocity corrections from forces added after post_force
  // This handles stochastic forces from Langevin thermostats
  
  atomKK->sync(execution_space, V_MASK | F_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK);
  atomKK->modified(execution_space, V_MASK);
  
  auto v_current = atomKK->k_v.view<DeviceType>();
  auto f_current = atomKK->k_f.view<DeviceType>();
  auto rmass_view = atomKK->k_rmass.view<DeviceType>();
  auto mass_view = atomKK->k_mass.view<DeviceType>();
  auto type_view = atomKK->k_type.view<DeviceType>();
  auto mask_view = atomKK->k_mask.view<DeviceType>();
  
  int nlocal = atomKK->nlocal;
  if (igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;

  double dtf = update->dt * force->ftm2v;
  int groupbit_copy = groupbit;
  auto f_pre_copy = f_pre_kk;
  bool use_rmass = rmass_view.data() != nullptr;

  // Apply force corrections using Kokkos parallel operation
  Kokkos::parallel_for(
      nlocal,
      KOKKOS_LAMBDA(int i) {
          if (mask_view[i] & groupbit_copy) {
              double mass_i = use_rmass ? rmass_view[i] : mass_view[type_view[i]];
              double dtfm = dtf / mass_i;
              
              // Apply only the incremental force (f - f_pre) to velocities
              v_current(i, 0) += (f_current(i, 0) - f_pre_copy(i, 0)) * dtfm;
              v_current(i, 1) += (f_current(i, 1) - f_pre_copy(i, 1)) * dtfm;
              v_current(i, 2) += (f_current(i, 2) - f_pre_copy(i, 2)) * dtfm;
          }
      }
  );
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class FixMetatomicKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixMetatomicKokkos<LMPHostType>;
#endif
}
