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

#include "metatomic_units.h"

#include <map>
#include <string>

namespace LAMMPS_NS {

const std::map<std::string, std::string> quantity_map = {
    {"energy", "energy"},
    {"energy_ensemble", "energy"},
    {"energy_uncertainty", "energy"},
    {"forces", "force"},
    {"non_conservative_forces", "force"},
    {"stress", "stress"},
    {"non_conservative_stress", "stress"},
    {"masses", "mass"},
    {"position", "position"},
    {"momenta", "momentum"},
    {"velocities", "velocity"},
    {"charges", "charge"},
    {"heat_flux", "heat_flux"},
};

const std::map<std::string, std::map<std::string, std::string>> unit_map = {
    {"mass", {
        {"real", "g/mol"},
        {"metal", "g/mol"},
        {"si", "kg"},
        {"cgs", "g"},
        {"electron", "u"},
        {"micro", "pg"},
        {"nano", "ag"}
    }},
    {"position", {
        {"real", "A"},
        {"metal", "A"},
        {"si", "m"},
        {"cgs", "cm"},
        {"electron", "Bohr"},
        {"micro", "micrometer"},
        {"nano", "nanometer"}
    }},
    {"energy", {
        {"real", "kcal/mol"},
        {"metal", "eV"},
        {"si", "J"},
        {"cgs", "erg"},
        {"electron", "Hartree"},
        {"micro", "pg*micrometer^2/microsecond^2"},
        {"nano", "ag*nm^2/ns^2"}
    }},
    {"velocity", {
        {"real", "A/fs"},
        {"metal", "A/ps"},
        {"si", "m/s"},
        {"cgs", "cm/s"},
        {"electron", "Bohr*Hartree/hbar"},
        {"micro", "micrometer/microsecond"},
        {"nano", "nm/ns"}
    }},
    {"force", {
        {"real", "kcal/mol/A"},
        {"metal", "eV/A"},
        {"si", "kg*m/s^2"},
        {"cgs", "g*cm/s^2"},
        {"electron", "Hartree/Bohr"},
        {"micro", "pg*micrometer/microsecond^2"},
        {"nano", "ag*nm/ns^2"}
    }},
    {"charge", {
        {"real", "e"},
        {"metal", "e"},
        {"si", "C"},
        {"cgs", "esu"},
        {"electron", "e"},
        {"micro", "pC"},
        {"nano", "e"}
    }},
    {"lmp::density", {
        {"real", ""},
        {"metal", ""},
        {"si", ""},
        {"cgs", ""},
        {"electron", ""},
        {"micro", ""},
        {"nano", ""}
    }},
    {"lmp::dipole_moment", {}},
    {"lmp::dynamic_viscosity", {}},
    {"lmp::electric_field", {}},
    {"lmp::pressure", {}},
    {"lmp::temperature", {}},
    {"lmp::time", {}},
    {"lmp::torque", {}},
};
// simple struct to hold unit conversion factors for a given unit style
}    // namespace LAMMPS_NS
