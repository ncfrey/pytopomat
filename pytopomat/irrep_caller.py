"""
Interface to irrep.

"""

import warnings
import os
from os import path
import subprocess
import re

from monty.json import MSONable
from monty.dev import requires
from monty.os.path import which
from monty.serialization import loadfn

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure

import numpy as np

__author__ = "Jason Munro, Nathan C. Frey"
__copyright__ = "MIT License"
__version__ = "0.0.1"
__maintainer__ = "Jason Munro, Nathan C. Frey, "
__email__ = "jmunro@lbl.gov, ncfrey@lbl.gov,"
__status__ = "Development"
__date__ = "Feb 2021"

IRREPEXE = which("irrep")


class IrrepCaller:
    @requires(
        IRREPEXE,
        "IRREPCaller requires irrep to be in the path.\n"
        "Please follow the instructions in https://pypi.org/project/irrep and https://arxiv.org/pdf/2009.01764.pdf ",
    )
    def __init__(self, folder_name, code="vasp", enable_spinor=True, add_args={}):
        """
        Run irrep to compute irreducible representations (irreps) of electronic states from wavefunctions and
        symmetry operations determined from an input structure.

        For running with vasp (default), it requires a calculation with LWAVE=.TRUE. This does NOT use the symmetry 
        operations found in the OUTCAR file.

        Something like "phonopy --tolerance 0.01 --symmetry -c POSCAR" should be used to ensure
        the crystal is in a standard setting before the calculation.

        Args:
            folder_name (str): Path to directory with input data at kpts where irreps should be computed.
            code (str): The code to run with. Default is vasp.
            enable_spinor (bool): Whether to include the '-spinor' flag when calling irrep.
            add_args (dict): Dictionary of additional arguments (i.e. {-Ecut: 50}).
        """

        # Check for POSCAR and WAVECAR if using vasp
        if code == "vasp":
            if not path.isfile(folder_name + "/WAVECAR") or not path.isfile(
                folder_name + "/POSCAR"
            ):
                raise FileNotFoundError()

        os.chdir(folder_name)

        # Call irrep
        cmd_list = ["irrep", "-code", code]

        if enable_spinor:
            cmd_list += ["-spinor"]

        for arg, val in add_args.items():
            cmd_list += [str(arg), str(val)]

        with open("outir.txt", "w") as out, open("err.txt", "w") as err:
            process = subprocess.Popen(cmd_list, stdout=out, stderr=err)

        process.communicate()  # pause while irrep is executing

        self.output = None

        # Process output
        if path.isfile("outir.txt"):
            self.output = IrrepOutput("outir.txt")

        else:
            raise FileNotFoundError()


class IrrepOutput(MSONable):
    def __init__(
        self,
        irrep_output,
        efermi=None,
        saved_bands=None,
        starting_band=None,
        energy_cutoff=None,
        spacegroup_no=None,
        spin_polarized=None,
        parity_eigenvals=None,
    ):
        """
        This class processes results from irrep to get irreps of electronic states. 

        Refer to https://arxiv.org/pdf/2009.01764.pdf for further explanation of parameters.

        Args:
            irvsp_output (txt file): output from irvsp.
            efermi (float): Supplied fermi energy used to shift band energies.
            saved_bands (int): Number of bands saved in output.
            starting_band (int): Starting band number.
            energy_cutoff (int): Plane-wave energy cutoff in eV used to generate g-vectors.
            spacegroup_no (int): Space group number detected.
            parity_eigenvals (dict): band index, band degeneracy, energy eigenval, Re(parity eigenval)

        """

        self._irrep_output = irrep_output
        self.efermi = efermi
        self.saved_bands = saved_bands
        self.starting_band = starting_band
        self.energy_cutoff = energy_cutoff
        self.spacegroup_no = spacegroup_no
        self.spin_polarized = spin_polarized
        self.parity_eigenvals = parity_eigenvals

        self._parse_stdout(irrep_output)

    def _parse_stdout(self, irrep_output):

        try:
            with open(irrep_output, "r") as file:
                lines = file.readlines()

                for idx, line in enumerate(lines):
                    if "Efermi" in line:
                        e = line.split()[3]
                        self.efermi = float(e) if e != "None" else None
                        init_block_start = idx
                        break

                band_line = lines[init_block_start + 2].split()
                self.saved_bands = int(band_line[1])
                self.starting_band = int(band_line[5])

                encut_line = lines[init_block_start + 4].split()
                self.energy_cutoff = float(encut_line[5])

                trunc_lines = lines[init_block_start:]

                # Define TRIM labels in units of primitive reciprocal vectors
                trim_labels = ["gamma", "x", "y", "z", "s", "t", "u", "r"]
                trim_pts = [
                    (0.0, 0.0, 0.0),
                    (0.5, 0.0, 0.0),
                    (0.0, 0.5, 0.0),
                    (0.0, 0.0, 0.5),
                    (0.5, 0.5, 0.0),
                    (0.0, 0.5, 0.5),
                    (0.5, 0.0, 0.5),
                    (0.5, 0.5, 0.5),
                ]

                trim_dict = {pt: label for (pt, label) in zip(trim_pts, trim_labels)}

                # Dicts with kvec index as keys
                parity_eigenvals = {}

                # Start of symm. op. and irrep trace info
                for idx, line in enumerate(trunc_lines):
                    if "INFORMATION ABOUT THE SPACE GROUP" in line:
                        symm_block_start = idx
                        break

                sg_line = trunc_lines[symm_block_start + 3].split()
                self.spacegroup_no = sg_line[3]

                # Find inv symm. op.
                for idx, line in enumerate(trunc_lines):
                    if "angle =  0 , inversion : True" in line:
                        inv_num = int(trunc_lines[idx + 1].split()[1]) - 1

                    if "k-point   1" in line:
                        irrep_block_start = idx
                        break

                # Find parity eigenvalues
                trace_start = False
                for line in trunc_lines[irrep_block_start:]:

                    if line.startswith("k-point"):  # New kvec
                        line_list = re.findall("\[(.*)\]", line)[0].split()
                        kvec = tuple([float(i) for i in line_list])
                        trim_label = trim_dict[kvec]

                        ndgs = []
                        bnd_evs = []
                        inv_evs = []

                    if "Energy  | multiplicity | irreps | sym. operations" in line:
                        trace_start = True
                        col_check = False
                        continue

                    if trace_start:
                        if "|" in line and not col_check:
                            inv_col = line.split().index(str(inv_num)) - 3
                            col_check = True
                        elif "|" in line:
                            line_list = line.split()

                            bnd_evs.append(float(line_list[0]))
                            ndgs.append(int(line_list[2]))
                            inv_ev = complex(line_list[inv_col + 6])
                            if inv_ev.imag != 0.0:
                                warnings.warn("Found complex parity eigenvalues!")
                                inv_evs.append(inv_ev)
                            else:
                                if not np.isclose(
                                    inv_ev.real % 1.0, 0.0, rtol=0, atol=0.03
                                ) or not np.isclose(
                                    inv_ev.real % 1.0, 1.0, rtol=0, atol=0.03
                                ):
                                    warnings.warn(
                                        "Irrep output data has non-integer parity eigenvalues!"
                                    )
                                inv_evs.append(inv_ev.real)

                        else:
                            trace_start = False
                            kvec_data = {
                                "band_degeneracy": ndgs,
                                "band_eigenval": bnd_evs,
                                "inversion_eigenval": inv_evs,
                            }

                            parity_eigenvals[trim_label] = kvec_data

                            continue

            self.parity_eigenvals = parity_eigenvals

        except FileNotFoundError:
            warnings.warn(
                "Irrep output not found. Setting instance attributes from direct inputs!"
            )

