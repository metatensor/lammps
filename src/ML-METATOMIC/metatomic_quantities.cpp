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

#include "metatomic_quantities.h"

#include <map>
#include <string>

namespace LAMMPS_NS {

const std::map<std::string, std::map<std::string, std::string>> metatomic_unit_map = {
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
    {"position", {  // is called "distance" in LAMMPS documentation
        {"real", "A"},
        {"metal", "A"},
        {"si", "m"},
        {"cgs", "cm"},
        {"electron", "Bohr"},
        {"micro", "micrometer"},
        {"nano", "nanometer"}
    }},

    // stardard quantities in metatomic, but not in LAMMPS:
    {"stress", { // equals to the negative of pressure
        {"real", "atm"},
        {"metal", "bar"},
        {"si", "Pa"},
        {"cgs", "10^-6 * bar"},
        {"electron", "Pa"},
        {"micro", "pg/micrometer/microsecond^2"},
        {"nano", "ag/ns^2/nm"}
    }},
    {"non_conservative_force", {
        {"real", "kcal/mol/A"},
        {"metal", "eV/A"},
        {"si", "kg*m/s^2"},
        {"cgs", "g*cm/s^2"},
        {"electron", "Hartree/Bohr"},
        {"micro", "pg*micrometer/microsecond^2"},
        {"nano", "ag*nm/ns^2"}
    }},
    {"non_conservative_stress", {
        {"real", "atm"},
        {"metal", "bar"},
        {"si", "Pa"},
        {"cgs", "10^-6 * bar"},
        {"electron", "Pa"},
        {"micro", "pg/micrometer/microsecond^2"},
        {"nano", "ag/ns^2/nm"}
    }},
    {"momentum", { // mass * velocity, see group.cpp:1186-1188
        {"real", "g/mol*A/fs"},
        {"metal", "g/mol*A/ps"},
        {"si", "kg*m/s"},
        {"cgs", "g*cm/s"},
        {"electron", "u*Bohr*Hartree/hbar"},
        {"micro", "pg*micrometer/microsecond"},
        {"nano", "ag*nm/ns"}
    }},
    {"heat_flux", {  // energy * velocity, see compute_heat_flux.cpp:136-138
        {"real", "kcal/mol*A/fs"},
        {"metal", "eV*A/ps"},
        {"si", "J*m/s"},
        {"cgs", "erg*cm/s"},
        {"electron", "Bohr*Hartree^2/hbar"},
        {"micro", "pg*micrometer^3/microsecond^3"},
        {"nano", "ag*nm^3/ns^3"}
    }}
};

const std::map<std::string, std::map<std::string, std::string>> metatomic_quantity_shape = {
    {"energy", {{"shape", "scalar"}, {"size", ""}}},
    {"force", {{"shape", "vector"}, {"size", "3"}}},
    {"stress", {{"shape", "vector"}, {"size", "9"}}},
    {"non_conservative_force", {{"shape", "vector"}, {"size", "3"}}},
    {"non_conservative_stress", {{"shape", "vector"}, {"size", "9"}}},
    {"heat_flux", {{"shape", "vector"}, {"size", "3"}}},
    {"mass", {{"shape", "scalar"}, {"size", ""}}},
    {"position", {{"shape", "vector"}, {"size", "3"}}},
    {"velocity", {{"shape", "vector"}, {"size", "3"}}},
    {"charge", {{"shape", "scalar"}, {"size", ""}}},
    {"momentum", {{"shape", "vector"}, {"size", "3"}}}
};
}    // namespace LAMMPS_NS
