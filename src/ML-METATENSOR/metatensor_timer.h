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

#ifndef LMP_METATENSOR_TIMER_H
#define LMP_METATENSOR_TIMER_H

#include <iostream>
#include <string>
#include <chrono>

namespace LAMMPS_NS {

struct ScopeTimer {
  ScopeTimer(std::string name):
    name_(std::move(name)),
    start_(std::chrono::high_resolution_clock::now()) {}

  ~ScopeTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
    std::cout << name_ << " took " << time_ns / 1e6 << "ms" << std::endl;
  }

  std::string name_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

}    // namespace LAMMPS_NS

#endif
