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

namespace LAMMPS_NS {

template<class DeviceType>
class MetatensorSystemAdaptorKokkos;

template<class DeviceType>
struct PairMetatensorDataKokkos;

/// I noticed that most other kokkos packages inherit from their non-kokkos
/// counterparts. It doesn't look like a good idea to me because
/// they end up overriding everything... Not doing it here for now.
template<class DeviceType>
class PairMetatensorKokkos : public Pair {
public:
    PairMetatensorKokkos(class LAMMPS *);
    ~PairMetatensorKokkos();

    void compute(int, int) override;
    void settings(int, char **) override;
    void coeff(int, char **) override;
    void init_style() override;
    double init_one(int, int) override;
    void init_list(int id, NeighList *ptr) override;

    void allocate();
private:
    PairMetatensorDataKokkos<DeviceType>* mts_data;

    // mapping from LAMMPS types to metatensor types
    int32_t* type_mapping;
};

}    // namespace LAMMPS_NS

#endif
#endif
