/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(metatomic,FixMetatomic);
// clang-format on
#else

#ifndef LMP_FIX_FLASHMD_H
#define LMP_FIX_FLASHMD_H

#include "fix.h"

#include <metatomic/torch.hpp>

namespace LAMMPS_NS {
class MetatomicSystemAdaptor;
class PairMetatomicData;

class FixMetatomic : public Fix {
 public:
  FixMetatomic(class LAMMPS *, int, char **);
  ~FixMetatomic();

  int setmask() override;
  void init() override;
  std::vector<torch::DeviceType> available_devices();
  void pick_device(torch::Device* device, const char* requested);
  
  // Integration methods for ML-driven dynamics
  void initial_integrate(int) override;  // ML prediction of positions/momenta
  void post_force(int) override;         // Snapshot forces for Langevin compatibility
  void final_integrate() override;       // Apply force corrections
  void init_list(int id, NeighList *ptr) override;

 protected:
  double dt;                    // Timestep
  std::string model_path;       // Path to ML model file
  std::string requested_device; // Device to run model on (cpu/cuda/mps)
   
  // Metatomic model data and configuration
  PairMetatomicData* mta_data;
  NeighList *mta_list;
  int mta_list_reqid;

  // Force snapshot for Langevin compatibility
  // Stores forces at post_force() time to isolate stochastic contributions
  double **f_pre = nullptr;
  void ensure_capacity();  // Ensures f_pre has sufficient capacity
  int nmax = 0;            // Current allocated size of f_pre

  // Mapping from LAMMPS atom types to metatomic model types
  int32_t *type_mapping;
  
  // Helper class to convert between LAMMPS and metatomic representations
  std::unique_ptr<MetatomicSystemAdaptor> system_adaptor;
};

}    // namespace LAMMPS_NS

#endif
#endif
