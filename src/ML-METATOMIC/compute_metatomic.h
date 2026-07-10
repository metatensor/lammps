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
#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(metatomic, ComputeMetatomic);
// clang-format on
#else

#ifndef LMP_COMPUTE_METATOMIC_H
#define LMP_COMPUTE_METATOMIC_H

#include "compute.h"

#include <optional>

namespace c10 {
    class Device;
    enum class DeviceType: int8_t;
}

namespace at {
    class Tensor;
}

namespace LAMMPS_NS {
class MetatomicSystemAdaptor;
struct ComputeMetatomicData;

class ComputeMetatomic : public Compute {
public:
    ComputeMetatomic(class LAMMPS *, int, char **);
    ~ComputeMetatomic();
    void init() override;
    void init_list(int id, NeighList *ptr) override;
    double compute_scalar() override;
    void compute_vector() override;
    void compute_peratom() override;

protected:
    std::string model_path;
    std::optional<std::string> requested_device;
    std::optional<std::string> extensions_directory;
    std::string output_name;

    // Metatomic model data
    ComputeMetatomicData* mta_data;
    NeighList *mta_list;

    // Mapping from LAMMPS atom types to metatomic model types
    int32_t *type_mapping;

    // Helper class to convert between LAMMPS and metatomic representations
    std::unique_ptr<MetatomicSystemAdaptor> system_adaptor;

private:
    // Cache the output of the most recent model evaluation
    void compute();
    enum ResultKind {
        RESULT_NONE,
        RESULT_GLOBAL_SCALAR,
        RESULT_GLOBAL_VECTOR,
        RESULT_PERATOM_SCALAR,
        RESULT_PERATOM_VECTOR
    };
    bool layout_initialized = false;
    ResultKind result_kind = RESULT_NONE;
    bigint last_eval_step = -1;
    std::vector<int> local_indices_cache_;
    double scalar_cache_ = 0.0;
    std::vector<double> vector_cache_;
    std::vector<double> peratom_cache_;
    std::vector<double> peratom_array_cache_;
    int n_max_atoms = 0;
    int n_samples = 0;
    int n_components = 0;
    int n_properties = 0;
    void check_or_init_layout(ResultKind kind, int size, int peratom_cols);
    void clear_cache();
    const char *result_kind_name(ResultKind kind);
};

}    // namespace LAMMPS_NS

#endif
#endif
