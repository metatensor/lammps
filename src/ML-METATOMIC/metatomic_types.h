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

#include "lammps.h"

#include <map>
#include <string>

#include <torch/torch.h>
#include <metatensor/torch.hpp>
#include <metatomic/torch.hpp>


#ifndef LMP_METATOMIC_TYPES_H
#define LMP_METATOMIC_TYPES_H

namespace LAMMPS_NS {

class Atom;

struct CommonMetatomicData {
   CommonMetatomicData(std::string length_unit);
   void load_model(LAMMPS* lmp, const char* path, const char* extensions_directory);

   // pick the compute device from the model's supported devices and the user
   // request (or nullptr), storing the result in `this->device`. `cmd_name` is
   // used as a prefix in error messages (e.g. "pair_style metatomic").
   void pick_device(LAMMPS* lmp, const char* requested, const char* cmd_name);

   // resolve the torch dtype (float32/float64) requested by the model
   c10::ScalarType model_dtype(LAMMPS* lmp) const;

   // compute `this->max_cutoff` from the model's interaction range, falling
   // back to the requested neighbor lists when the range is infinite
   void resolve_max_cutoff(LAMMPS* lmp);

   // set the selected atoms in the evaluation options according to the request of lmp,
   // build selected atoms on CPU, then copy to device
   void set_selected_atoms(Atom* atom, int groupbit);

   // collect the additional inputs requested by the model
   std::map<std::string, metatomic_torch::ModelOutput> collect_requested_inputs() const;

   // the metatomic model
   std::unique_ptr<metatensor_torch::Module> model;
   // the path used to load the model
   std::string model_path;
   // device to use for the calculations
   torch::Device device;
   // model capabilities, declared by the model
   metatomic_torch::ModelCapabilities capabilities;
   // run-time evaluation options, decided by this class
   metatomic_torch::ModelEvaluationOptions evaluation_options;

   // should metatomic check the data LAMMPS send to the model
   // and the data the model returns?
   bool check_consistency = false;
   // how far away the model needs to know about neighbors
   double max_cutoff = -1;

   // allocation cache for the selected atoms (device)
   torch::Tensor selected_atoms_values;
   // allocation cache for the selected atoms (CPU)
   torch::Tensor selected_atoms_values_cpu;
};

struct PairMetatomicData: public CommonMetatomicData {
   PairMetatomicData(std::string length_unit): CommonMetatomicData(std::move(length_unit)) {}

   // the energy output we'll request from a model
   metatomic_torch::ModelOutput energy_output;
   // wether the model capabilities say that it can do per-atom energies
   bool is_energy_output_per_atom = false;

   // energy uncertainty output we'll request from a model, or nullptr if the
   // model does not have such output
   metatomic_torch::ModelOutput uncertainty_output;
   // threshold for energy uncertainty warnings
   double uncertainty_threshold = 0.0;

   // non-conservative forces/stress outputs we'll request from a model, or
   // nullptr if the model does not have such outputs
   metatomic_torch::ModelOutput nc_forces_output;
   metatomic_torch::ModelOutput nc_stress_output;

   // which non-conservative outputs to use
   enum class NonConservativeMode { OFF, ON, FORCES, STRESS };
   NonConservativeMode non_conservative = NonConservativeMode::OFF;

   // energy key for the model
   std::string energy_key;
   // energy uncertainty key for the model
   std::string energy_uq_key;
   // non-conservative forces key for the model
   std::string nc_forces_key;
   // non-conservative stress key for the model
   std::string nc_stress_key;
};

struct FixMetatomicData: public CommonMetatomicData {
   FixMetatomicData(std::string length_unit): CommonMetatomicData(std::move(length_unit)) {}
};

struct ComputeMetatomicData: public CommonMetatomicData {
   ComputeMetatomicData(std::string length_unit): CommonMetatomicData(std::move(length_unit)) {}
   // the inputs the model requested, and their corresponding holders
   std::map<std::string, metatomic_torch::ModelOutput> requested_inputs;
   // the output we'll request from a model
   metatomic_torch::ModelOutput requested_output;
};

}    // namespace LAMMPS_NS

#endif
