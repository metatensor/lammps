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

#include "fix_flashmd.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "update.h"

#include <iostream>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixFlashMD::FixFlashMD(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  std::string energy_unit;
  std::string length_unit;
  if (strcmp(update->unit_style, "real") == 0) {
      length_unit = "angstrom";
      energy_unit = "kcal/mol";
  } else if (strcmp(update->unit_style, "metal") == 0) {
      length_unit = "angstrom";
      energy_unit = "eV";
  } else if (strcmp(update->unit_style, "si") == 0) {
      length_unit = "meter";
      energy_unit = "joule";
  } else if (strcmp(update->unit_style, "electron") == 0) {
      length_unit = "Bohr";
      energy_unit = "Hartree";
  } else {
      error->all(FLERR, "unsupported units '{}' for fix flashmd ", update->unit_style);
  }

  if (narg < 4) error->all(FLERR, "Illegal fix flashmd command");

  bool types_are_set = false;
  std::string model_path = arg[3];
  std::string energy_model_path;
  std::string requested_device;
  bool rescale_energy = false;

  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "types") == 0) {
      types_are_set = true;
      // try to run std::stoi on all the following arguments; if an exception is thrown,
      // we stop parsing the types
      int current_num_types = 0;
      iarg++;
      while (iarg < narg) {
        int type = -1;
        try {
          type = std::stoi(arg[iarg]);
          iarg++;
        } catch (const std::invalid_argument &) {
          break;  // stop parsing types on invalid argument to std::stoi
        }
        if (type <= 0) {
          error->all(FLERR, "Illegal fix flashmd command: type {} should be > 0", type);
        }
        current_num_types++;
        if (current_num_types > atom->ntypes) {
          error->all(FLERR, "Illegal fix flashmd command: too many types specified");
        }
      }
    } else if (strcmp(arg[iarg], "energy") == 0) {
      if (iarg + 1 > narg) error->all(FLERR, "Illegal fix flashmd command");
      energy_model_path = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "device") == 0) {
      if (iarg + 1 > narg) error->all(FLERR, "Illegal fix flashmd command");
      requested_device = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "rescale_energy") == 0) {
      if (iarg + 1 > narg) error->all(FLERR, "Illegal fix flashmd command");
      if (strcmp(arg[iarg + 1], "on") == 0) {
        rescale_energy = true;
      } else if (strcmp(arg[iarg + 1], "off") == 0) {
        rescale_energy = false;
      } else {
        error->all(FLERR, "Illegal fix flashmd command: expected 'on' or 'off' after 'rescale_energy'");
      }
      iarg += 2;
    } else {
      error->all(FLERR, "Illegal fix flashmd command");
    }
  }
  if (!types_are_set) {
    error->all(FLERR, "Illegal fix flashmd command: no types specified");
  }

  std::cout << "types_are_set = " << types_are_set << std::endl;
  std::cout << "energy_model_path = " << energy_model_path << std::endl;
  std::cout << "model_path = " << model_path << std::endl;
  std::cout << "requested_device = " << requested_device << std::endl;


  exit(1);

  time_integrate = 1;  // this tells LAMMPS that this fix advances simulation time
  // Note: for now we don't allow dynamic groups (dynamic_group_allow variable)
}

/* ---------------------------------------------------------------------- */

int FixFlashMD::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  // mask |= FINAL_INTEGRATE;  // ??
  // mask |= INITIAL_INTEGRATE_RESPA;  // ??
  // mask |= FINAL_INTEGRATE_RESPA;  // ??
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixFlashMD::init()
{
  dt = update->dt;
  // TODO: what to do with units if not metal????

  // Load model here

  // Initialize metatensor system object here?
}

void FixFlashMD::initial_integrate(int /*vflag*/)
{
  double dtfm;

  // update v and x of atoms in group

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // if (rmass) {
  //   for (int i = 0; i < nlocal; i++)
  //     if (mask[i] & groupbit) {
  //       dtfm = dtf / rmass[i];
  //       v[i][0] += dtfm * f[i][0];
  //       v[i][1] += dtfm * f[i][1];
  //       v[i][2] += dtfm * f[i][2];
  //       x[i][0] += dtv * v[i][0];
  //       x[i][1] += dtv * v[i][1];
  //       x[i][2] += dtv * v[i][2];
  //     }

  // } else {
  //   for (int i = 0; i < nlocal; i++)
  //     if (mask[i] & groupbit) {
  //       dtfm = dtf / mass[type[i]];
  //       v[i][0] += dtfm * f[i][0];
  //       v[i][1] += dtfm * f[i][1];
  //       v[i][2] += dtfm * f[i][2];
  //       x[i][0] += dtv * v[i][0];
  //       x[i][1] += dtv * v[i][1];
  //       x[i][2] += dtv * v[i][2];
  //     }
  // }
}
