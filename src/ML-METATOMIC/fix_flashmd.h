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
FixStyle(flashmd,FixFlashMD);
// clang-format on
#else

#ifndef LMP_FIX_FLASHMD_H
#define LMP_FIX_FLASHMD_H

#include "fix.h"

#include <metatomic/torch.hpp>

namespace LAMMPS_NS {
class MetatomicSystemAdaptor;
class PairMetatomicData;

class FixFlashMD : public Fix {
 public:
  FixFlashMD(class LAMMPS *, int, char **);
  ~FixFlashMD();

  int setmask() override;
  void init() override;
  std::vector<torch::DeviceType> available_devices();
  void pick_device(torch::Device* device, const char* requested);
  void initial_integrate(int) override;
  void init_list(int id, NeighList *ptr) override;

 protected:
  double dt;
  std::string model_path;
  std::string requested_device;
   
  PairMetatomicData* mta_data;
  NeighList *mta_list;
  int mta_list_reqid;

  // mapping from LAMMPS types to metatomic types
  int32_t *type_mapping;
  // Helper class to convert between LAMMPS and metatomic.
  std::unique_ptr<MetatomicSystemAdaptor> system_adaptor;
};

}    // namespace LAMMPS_NS

#endif
#endif
