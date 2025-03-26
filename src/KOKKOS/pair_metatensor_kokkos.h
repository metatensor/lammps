/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS Development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
#ifdef PAIR_CLASS
// clang-format off
PairStyle(metatensor/kk, PairMetatensorKokkos<LMPDeviceType>);
// clang-format on
#else

#ifndef LMP_PAIR_METATENSOR_KOKKOS_H
#define LMP_PAIR_METATENSOR_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_metatensor.h"

namespace LAMMPS_NS {

template<class DeviceType>
class MetatensorSystemAdaptorKokkos;

template<class DeviceType>
struct PairMetatensorDataKokkos;

/// I noticed that most other kokkos packages inherit from their non-kokkos
/// counterparts. It doesn't look like a good idea to me because
/// they end up overriding everything... Not doing it here for now.
template<class DeviceType>
class PairMetatensorKokkos : public PairMetatensor {
public:
    PairMetatensorKokkos(class LAMMPS *);
    ~PairMetatensorKokkos();

    void init_style() override;
    void compute(int eflag, int vflag) override;
private:
    void pick_device(c10::Device* device, const char* requested) override;

    Kokkos::View<int32_t*, Kokkos::LayoutRight, DeviceType> type_mapping_kk;
};

}    // namespace LAMMPS_NS

#endif
#endif
