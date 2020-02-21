import warnings
import os
from os import path
import subprocess

from monty.json import MSONable
from monty.dev import requires
from monty.os.path import which
from monty.serialization import loadfn

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure

"""
This module offers a high level framework for analyzing topological materials in a 
high-throughput context with VASP, Z2Pack, irvsp, and Vasp2Trace.

"""

__author__ = "Nathan C. Frey, Jason Munro"
__copyright__ = "MIT License"
__version__ = "0.0.1"
__maintainer__ = "Nathan C. Frey, Jason, Munro"
__email__ = "ncfrey@lbl.gov, jmunro@lbl.gov"
__status__ = "Development"
__date__ = "August 2019"

IRVSPEXE = which("irvsp")


class IRVSPCaller:
    @requires(
        IRVSPEXE,
        "IRVSPCaller requires irvsp to be in the path.\n"
        "Please follow the instructions in https://arxiv.org/pdf/2002.04032.pdf\n"
        "https://github.com/zjwang11/irvsp/blob/master/src_irvsp_v2.tar.gz",
    )
    def __init__(self, folder_name):
        """
        Run irvsp to compute irreducible representations (irreps) of electronic states from wavefunctions (WAVECAR) and
        symmetry operations (OUTCAR).

        Requires a calculation with ISYM=1,2 and LWAVE=.TRUE.

        Something like "phonopy --tolerance 0.01 --symmetry -c POSCAR" should be used to ensure
        the crystal is in a standard setting before the calculation.

        irvsp v2 is needed to handle all 230 space groups (including nonsymmorphic sgs).

        Args:
            folder_name (str): Path to directory with POSCAR, OUTCAR and WAVECAR at kpts where irreps should be computed.
        """

        # Check for OUTCAR and WAVECAR
        if (
            not path.isfile(folder_name + "/OUTCAR")
            or not path.isfile(folder_name + "/WAVECAR")
            or not path.isfile(folder_name + "/POSCAR")
        ):
            raise FileNotFoundError()

        os.chdir(folder_name)

        # Get sg number of structure
        s = Structure.from_file("POSCAR")
        sga = SpacegroupAnalyzer(s, symprec=0.01)
        sgn = sga.get_space_group_number()
        v = 1  # version 1 of irvsp, symmorphic symmetries

        # Check if symmorphic (same symm elements as corresponding point group)
        # REF: http://kuchem.kyoto-u.ac.jp/kinso/weda/data/group/space.pdf
        fpath = os.path.join(os.path.dirname(__file__), "symmorphic_spacegroups.json")
        ssgs = loadfn(fpath)["ssgs"]

        # Remove SGOs from OUTCAR other than identity and inversion to avoid errors
        if sgn not in ssgs:  # non-symmorphic
            self.modify_outcar()
            sgn = 2  # SG 2 (only E and I)

        # Call irvsp
        cmd_list = ["irvsp", "-sg", str(sgn), "-v", str(v)]
        with open("outir.txt", "w") as out, open("err.txt", "w") as err:
            process = subprocess.Popen(cmd_list, stdout=out, stderr=err)

        process.communicate()  # pause while irvsp is executing

        self.output = None

        # Process output
        if path.isfile("outir.txt"):
            self.output = IRVSPOutput("outir.txt")

        else:
            raise FileNotFoundError()

    @staticmethod
    def modify_outcar(name="OUTCAR.bkp"):
        """
        For a non-symmorphic material, delete all space group ops from OUTCAR except for identity (E)
        and inversion (I). This allows the command "irvsp -sg 2 -v 1" to compute only I eigenvalues.

        Must be run in a directory with OUTCAR.

        Args:
            name (str): Name for unmodified copy of OUTCAR.
        """

        # Check for OUTCAR and WAVECAR
        if not path.isfile("OUTCAR"):
            raise FileNotFoundError()

        sgo_lines = []  # OUTCAR lines with superfluous SGOs

        # Write a temp file without the extra SGOs
        with open("OUTCAR", "r") as f:
            with open("temp.txt", "w") as output:
                lines = f.readlines()

                for idx, line in enumerate(lines):
                    if "INISYM" in line:
                        num_ops = int(line.strip().split(" ")[4])
                    if "irot" in line:  # Start of SGOs
                        sgo_lines = list(range(idx + 3, idx + num_ops + 1))
                    if idx not in sgo_lines:
                        output.write(line)

        os.rename("OUTCAR", name)
        os.rename("temp.txt", "OUTCAR")


class IRVSPOutput(MSONable):
    def __init__(
        self,
        irvsp_output,
        symmorphic=None,
        inversion=None,
        soc=None,
        spin_polarized=None,
        parity_eigenvals=None,
    ):
        """
        This class processes results from irvsp to get irreps of electronic states. 

        Refer to https://arxiv.org/pdf/2002.04032.pdf for further explanation of parameters.

        Args:
            irvsp_output (txt file): output from irvsp.
            symmorphic (Bool): Symmorphic space group?
            inversion (Bool): Centrosymmetric space group?
            soc (Bool): Spin-orbit coupling included?
            spin_polarized (Bool): Spin-polarized system?
            parity_eigenvals (dict): band index, band degeneracy, energy eigenval, Re(parity eigenval)

        """

        self._irvsp_output = irvsp_output

        self.symmorphic = symmorphic
        self.inversion = inversion
        self.soc = soc
        self.spin_polarized = spin_polarized
        self.parity_eigenvals = parity_eigenvals

        self._parse_stdout(irvsp_output)

    def _parse_stdout(self, irvsp_output):

        try:
            with open(irvsp_output, "r") as file:
                lines = file.readlines()

                # Get header info
                symm_line = lines[7]
                if "Non-symmorphic" in symm_line:
                    symmorphic = False
                else:
                    symmorphic = True

                if "without" in symm_line:
                    inversion = False
                else:
                    inversion = True

                soc_line = lines[9]
                if "No" in soc_line:
                    soc = False
                else:
                    soc = True

                sp_line = lines[10]
                if "No" in sp_line:
                    spin_polarized = False
                else:
                    spin_polarized = True

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

                # Start of irrep trace info
                for idx, line in enumerate(lines):
                    if line.startswith("**********************"):
                        block_start = idx + 1
                        break

                trace_start = False
                for idx, line in enumerate(lines[block_start:]):
                    if line.startswith("k = "):  # New kvec
                        line_list = line.split(" ")[2:]
                        kvec = tuple([float(i) for i in line_list])
                        trim_label = trim_dict[kvec]

                    if "bnd ndg" in line:  # find inversion symmop position
                        trace_start = True  # Start of block of traces
                        bnds, ndgs, bnd_evs, inv_evs = [], [], [], []
                        line_list = line.strip().split(" ")
                        symmops = [i for i in line_list if i]
                        inv_num = symmops.index("I") - 3  # subtract bnd, ndg, ev
                        num_ops = len(symmops) - 3  # subtract bnd, ndg, ev

                    if trace_start and "0" in line:  # full trace line, not a blank line
                        line_list = line[6:].strip()
                        line_list = line_list.split("=", 1)[
                            0
                        ]  # delete irrep label at end of line
                        line_list = [i for i in line_list.split(" ") if i]

                        # Check that trace line is complete, no ?? or errors
                        if len(line_list) == num_ops + 1:  # symmops + band eigenval
                            bnd = int(
                                [i for i in line[:3].split(" ") if i][0]
                            )  # band index
                            ndg = int(line[5])  # band degeneracy

                            evs = [i for i in line_list if i]
                            bnd_ev = float(evs[0])
                            inv_ev = evs[inv_num + 1]
                            inv_ev = float(inv_ev[:4])  # delete imaginary part
                            bnds.append(bnd)
                            ndgs.append(ndg)
                            bnd_evs.append(bnd_ev)
                            inv_evs.append(inv_ev)

                    if line.startswith("**********************"):  # end of block
                        trace_start = False
                        kvec_data = {
                            "band_index": bnds,
                            "band_degeneracy": ndgs,
                            "band_eigenval": bnd_evs,
                            "inversion_eigenval": inv_evs,
                        }
                        parity_eigenvals[trim_label] = kvec_data

            self.symmorphic = symmorphic
            self.inversion = inversion
            self.soc = soc
            self.spin_polarized = spin_polarized
            self.parity_eigenvals = parity_eigenvals
            
        except:
            warnings.warn(
                "irvsp output not found. Setting instance attributes from direct inputs!"
            )
