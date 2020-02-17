import warnings
import os
from os import path
import logging
import subprocess

import numpy as np

from monty.json import MSONable
from monty.dev import requires
from monty.os.path import which
from pymatgen.core.operations import SymmOp
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.dimensionality import (
    get_dimensionality_larsen,
    get_dimensionality_cheon,
    get_dimensionality_gorai,
)
from pymatgen.analysis.local_env import MinimumDistanceNN

"""
This module offers a high level framework for analyzing topological materials in a high-throughput context with VASP, Z2Pack, and Vasp2Trace.

"""

__author__ = "Nathan C. Frey, Jason Munro"
__copyright__ = "MIT License"
__version__ = "0.0.1"
__maintainer__ = "Nathan C. Frey, Jason, Munro"
__email__ = "ncfrey@lbl.gov, jmunro@lbl.gov"
__status__ = "Development"
__date__ = "August 2019"

VASP2TRACEEXE = which("vasp2trace")
VASP2TRACE2EXE = which("vasp2trace2") 


class Vasp2TraceCaller:
    @requires(
        VASP2TRACEEXE,
        "Vasp2TraceCaller requires vasp2trace to be in the path."
        "Please follow the instructions at http://www.cryst.ehu.es/cgi-bin/cryst/programs/topological.pl.",
    )
    def __init__(self, folder_name):
        """
        Run vasp2trace to find the set of irreducible representations at each maximal k-vec of a space group, given the eigenvalues.

        vasp2trace requires a self-consistent VASP run with the flags ISTART=0 and ICHARG=2; followed by a band structure calculation with ICHARG=11, ISYM=2, LWAVE=.True.

        High-symmetry kpts that must be included in the band structure path for a given spacegroup can be found in the max_KPOINTS_VASP folder in the vasp2trace directory.

        Args:
            folder_name (str): Path to directory with OUTCAR and WAVECAR of band structure run with wavefunctions at the high-symmetry kpts.
        """

        # Check for OUTCAR and WAVECAR
        if not path.isfile(folder_name + "/OUTCAR") or not path.isfile(
            folder_name + "/WAVECAR"
        ):
            raise FileNotFoundError()

        # Call vasp2trace
        os.chdir(folder_name)
        process = subprocess.Popen(
            ["vasp2trace"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        stdout = stdout.decode()

        if stderr:
            stderr = stderr.decode()
            warnings.warn(stderr)

        if process.returncode != 0:
            raise RuntimeError(
                "vasp2trace exited with return code {}.".format(process.returncode)
            )

        self._stdout = stdout
        self._stderr = stderr
        self.output = None

        # Process output
        if path.isfile("trace.txt"):
            self.output = {}
            self.output["up"] = Vasp2TraceOutput("trace.txt")

        else:
            raise FileNotFoundError()


class Vasp2Trace2Caller:
    @requires(
        VASP2TRACE2EXE,
        "Vasp2TraceCaller requires vasp2trace2 to be in the path."
        "Please install from https://github.com/zjwang11/irvsp",
    )
    def __init__(self, folder_name):
        """
        Run vasp2trace_v2 to find the set of irreducible representations at each maximal k-vec of a space group, given the eigenvalues.

        version2 of vasp2trace is for spin-polarized calculations. The executable is renamed "vasp2trace2" to avoid conflict with v1.

        vasp2trace requires a self-consistent VASP run with the flags ISTART=0 and ICHARG=2; followed by a band structure calculation with ICHARG=11, ISYM=2, LWAVE=.True.

        High-symmetry kpts that must be included in the band structure path for a given spacegroup can be found in the max_KPOINTS_VASP folder in the vasp2trace directory.

        Args:
            folder_name (str): Path to directory with OUTCAR and WAVECAR of band structure run with wavefunctions at the high-symmetry kpts.
        """

        # Check for OUTCAR and WAVECAR
        if not path.isfile(folder_name + "/OUTCAR") or not path.isfile(
            folder_name + "/WAVECAR"
        ):
            raise FileNotFoundError()

        # Call vasp2trace
        os.chdir(folder_name)
        process = subprocess.Popen(
            ["vasp2trace2"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        stdout = stdout.decode()

        if stderr:
            stderr = stderr.decode()
            warnings.warn(stderr)

        if process.returncode != 0:
            raise RuntimeError(
                "vasp2trace2 exited with return code {}.".format(process.returncode)
            )

        self._stdout = stdout
        self._stderr = stderr
        self.output = None

        # Process spin-polarized output
        if path.isfile("trace_up.txt") and path.isfile("trace_dn.txt"):
            self.output = {}
            if path.isfile("trace_up.txt"):
                self.output["up"] = Vasp2TraceOutput("trace_up.txt")
            if path.isfile("trace_dn.txt"):
                self.output["down"] = Vasp2TraceOutput("trace_dn.txt")

        else:
            raise FileNotFoundError()


class Vasp2TraceOutput(MSONable):
    def __init__(
        self,
        vasp2trace_output,
        num_occ_bands=None,
        soc=None,
        num_symm_ops=None,
        symm_ops=None,
        num_max_kvec=None,
        kvecs=None,
        num_kvec_symm_ops=None,
        symm_ops_in_little_cogroup=None,
        traces=None,
    ):
        """
        This class processes results from vasp2trace to classify material band topology and give topological invariants.

        Refer to http://www.cryst.ehu.es/html/cryst/topological/File_Description.txt for further explanation of parameters.

        Args:
            vasp2trace_stdout (txt file): stdout from running vasp2trace.
            num_occ_bands (int): Number of occupied bands.
            soc (int): 0: no spin-orbit, 1: yes spin-orbit
            num_symm_ops (int): Number of symmetry operations.
            symm_ops (list): Each row is a symmetry operation (with spinor components if soc is enabled)
            num_max_kvec (int): Number of maximal k-vectors.
            kvecs (list): Each row is a k-vector.
            num_kvec_symm_ops (dict): {kvec_index: number of symm operations in the little cogroup of the kvec}. 
            symm_ops_in_little_cogroup (dict): {kvec_index: int indices that correspond to symm_ops}
            traces (dict): band index, band degeneracy, energy eigenval, Re eigenval, Im eigenval for each symm op in the little cogroup 

        """

        self._vasp2trace_output = vasp2trace_output

        self.num_occ_bands = num_occ_bands
        self.soc = soc
        self.num_symm_ops = num_symm_ops
        self.symm_ops = symm_ops
        self.num_max_kvec = num_max_kvec
        self.kvecs = kvecs
        self.num_kvec_symm_ops = num_kvec_symm_ops
        self.symm_ops_in_little_cogroup = symm_ops_in_little_cogroup
        self.traces = traces

        self._parse_stdout(vasp2trace_output)


    def _parse_stdout(self, vasp2trace_output):

        try:
            with open(vasp2trace_output, "r") as file:
                lines = file.readlines()

                # Get header info
                num_occ_bands = int(lines[0])
                soc = int(lines[1])  # No: 0, Yes: 1
                num_symm_ops = int(lines[2])
                symm_ops = np.ndarray.tolist(np.loadtxt(lines[3 : 3 + num_symm_ops]))
                num_max_kvec = int(lines[3 + num_symm_ops])
                kvecs = np.ndarray.tolist(
                    np.loadtxt(lines[4 + num_symm_ops : 4 + num_symm_ops + num_max_kvec])
                )

                # Dicts with kvec index as keys
                num_kvec_symm_ops = {}
                symm_ops_in_little_cogroup = {}
                traces = {}

                # Start of trace info
                trace_start = 5 + num_max_kvec + num_symm_ops
                start_block = 0  # Start of this block

                # Block start line #s
                block_starts = []
                for jdx, line in enumerate(lines[trace_start - 1 :], trace_start - 1):
                    # Parse input lines
                    line = [i for i in line.split(" ") if i]
                    if len(line) == 1:  # A single entry <-> new block
                        block_starts.append(jdx)

                # Loop over blocks of kvec data
                for idx, kpt in enumerate(kvecs):

                    start_block = block_starts[idx]
                    if idx < num_max_kvec - 1:
                        next_block = block_starts[idx + 1]
                        trace_str = lines[start_block + 2 : next_block]
                    else:
                        trace_str = lines[start_block + 2 :]

                    # Populate dicts
                    num_kvec_symm_ops[str(idx)] = int(lines[start_block])
                    soilcg = [
                        int(i.strip("\n"))
                        for i in lines[start_block + 1].split(" ")
                        if i.strip("\n")
                    ]
                    symm_ops_in_little_cogroup[str(idx)] = soilcg

                    trace = np.ndarray.tolist(np.loadtxt(trace_str))
                    traces[str(idx)] = trace
        except:
            warnings.warn(
                'Vasp2trace output not found. Setting instance attributes from direct inputs!')


class BandParity(MSONable):
    def __init__(self, v2t_output=None, trim_data=None, spin_polarized=None):
        """
        Determine parity of occupied bands at TRIM points with vasp2trace to calculate the Z2 topological invariant for centrosymmetric materials.

        Must give either v2t_output (non-spin-polarized) OR (up & down) for spin-polarized.

        Requires a VASP band structure run over the 8 TRIM points with:
        ICHARG=11; ISYM=2; LWAVE=.TRUE.

        This module depends on the vasp2trace script available in the path.
        Please download at http://www.cryst.ehu.es/cgi-bin/cryst/programs/topological.pl and consult the README.pdf for further help.

        If you use this module please cite:
        [1] Barry Bradlyn, L. Elcoro, Jennifer Cano, M. G. Vergniory, Zhijun Wang, C. Felser, M. I. Aroyo & B. Andrei Bernevig, Nature volume 547, pages 298–305 (20 July 2017).

        [2] M.G. Vergniory, L. Elcoro, C. Felser, N. Regnault, B.A. Bernevig, Z. Wang Nature (2019) 566, 480-485. doi:10.1038/s41586-019-0954-4.

        Args:
            v2t_output (dict): Dict of {'up': Vasp2TraceOutput object} or {'up': v2to, 'down': v2to}.

            trim_data (dict): Maps TRIM point labels to band eigenvals and energies.
                              Contains dict of {"energies": List, "iden": List, "parity": List}

            spin_polarized (bool): Spin-polarized or not.

        TODO:
            * Try to find a gapped subspace of Bloch bands
            * Compute overall parity and Z2=(v0, v1v2v3)
            * Report spin-polarized parity filters
        """

        self.v2t_output = v2t_output
        self.trim_data = trim_data
        self.spin_polarized = spin_polarized

        # Check if spin-polarized or not
        if "down" in v2t_output.keys():  # spin polarized
            self.spin_polarized = True
        else:
            self.spin_polarized = False

        if self.spin_polarized:
            self.trim_data = {}

            # Up spin
            parity_op_index_up = self._get_parity_op(
                self.v2t_output["up"].symm_ops)
            self.trim_data["up"] = self.get_trim_data(
                parity_op_index_up, self.v2t_output["up"]
            )

            # Down spin
            parity_op_index_dn = self._get_parity_op(
                self.v2t_output["down"].symm_ops)
            self.trim_data["down"] = self.get_trim_data(
                parity_op_index_dn, self.v2t_output["down"]
            )

        else:
            self.trim_data = {}

            parity_op_index = self._get_parity_op(
                self.v2t_output["up"].symm_ops)
            self.trim_data["up"] = self.get_trim_data(
                parity_op_index, self.v2t_output["up"]
            )

    @staticmethod
    def _get_parity_op(symm_ops):
        """Find parity in the list of SymmOps.

        Args:
            symm_ops (list): List of symmetry operations from v2t output.

        Returns:
            parity_op_index (int): Index of parity operator in the v2t output.

        """

        # x,y,z -> -x,-y,-z
        parity_mat = np.array([-1, 0, 0, 0, -1, 0, 0, 0, -1])

        parity_op_index = None



        for idx, symm_op in enumerate(symm_ops):
            try:
                rot_mat = symm_op[0:9]  # Rotation matrix
                trans_mat = symm_op[9:12]  # Translation matrix
            except TypeError:
                raise RuntimeError("No non-trivial symmetry operations in vasp2trace output!")

            # Find the parity operator
            if np.array_equal(rot_mat, parity_mat):
                parity_op_index = idx + 1  # SymmOps are 1 indexed

        if parity_op_index == None:
            raise RuntimeError("Parity operation not found in vasp2trace output!")
        else:
            return parity_op_index

    @staticmethod
    def get_trim_data(parity_op_index, v2t_output):
        """Tabulate parity and identity eigenvals, as well as energies for all 
            occupied bands over all TRIM points.

        Args:
            parity_op_index (int): Index of parity op in list of SymmOps.
            v2t_output (obj): V2TO object.

        Returns:
            trim_parities (dict): Maps TRIM label to band parities.

        """

        v2to = v2t_output

        # Check dimension of material and define TRIMs accordingly
        if v2to.num_max_kvec == 8:  # 3D

            # Define TRIM labels in units of primitive reciprocal vectors
            trim_labels = ["gamma", "x", "y", "z", "s", "t", "u", "r"]
            trim_pts = [
                (0, 0, 0),
                (0.5, 0, 0),
                (0, 0.5, 0),
                (0, 0, 0.5),
                (0.5, 0.5, 0),
                (0, 0.5, 0.5),
                (0.5, 0, 0.5),
                (0.5, 0.5, 0.5),
            ]

        elif v2to.num_max_kvec == 4:  # 2D
            trim_labels = ["gamma", "x", "y", "s"]
            trim_pts = [(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0.5, 0.5, 0)]

        trims = {
            trim_pt: trim_label for trim_pt, trim_label in zip(trim_pts, trim_labels)
        }

        trim_data = {trim_label: {'energies': [], 'iden': [], 'parity': []}
                     for trim_label in trim_labels}

        for idx, kvec in enumerate(v2to.kvecs):
            for trim_pt, trim_label in trims.items():
                if np.array_equal(kvec, trim_pt):  # Check if a TRIM

                    # Index of parity op for this kvec
                    kvec_parity_idx = v2to.symm_ops_in_little_cogroup[str(idx)].index(
                        parity_op_index
                    )

                    kvec_traces = v2to.traces[str(idx)]
                    for band_index, band in enumerate(kvec_traces):

                        # Real part of parity eigenval
                        band_parity_eigenval = band[3 + 2 * kvec_parity_idx]
                        band_iden_eigenval = band[3]
                        band_energy = band[2]

                        trim_data[trim_label]['parity'].append(
                            band_parity_eigenval)
                        trim_data[trim_label]['iden'].append(
                            band_iden_eigenval)
                        trim_data[trim_label]['energies'].append(band_energy)

        return trim_data

    def compute_z2(self, tol=2):
        """
        Compute Z2 topological indices (index) in 3D (2D) from TRIM band parities.

        Args:
            tol (float): Tolerance for average energy difference between bands at TRIM points to define independent band group.

        Returns:
            Z2 (list): List of integer Z2 indices (index) in 3D (2D). 
        """

        trim_labels = [key for key in self.trim_data["up"].keys()]

        trim_parities, trim_energies = self._format_parity_data()

        iband = self._get_band_subspace(tol=tol,
                                        trim_energies_formatted=trim_energies)
        print("Only considering last %i pairs of bands." % (iband-1))

        if len(trim_labels) == 8:
            Z2 = np.ones(4, dtype=int)

            for label in trim_labels:
                delta = 1
                for parity in trim_parities[label][:(-1*iband):-1]:
                    delta *= parity

                Z2[0] *= delta

                if label in ["x", "s", "u", "r"]:
                    Z2[1] *= delta
                if label in ["y", "s", "t", "r"]:
                    Z2[2] *= delta
                if label in ["z", "t", "u", "r"]:
                    Z2[3] *= delta

            return ((Z2 - 1) / -2) + 0

        elif len(trim_labels) == 4:
            Z2 = np.ones(1, dtype=int)

            for label in trim_labels:
                delta = 1
                for parity in trim_parities[label][:(-1*iband):-1]:
                    delta *= parity

                Z2 *= delta

            return ((Z2 - 1) / -2) + 0

        else:
            raise RuntimeError(
                "Incorrect number of k-points in vasp2trace output.")

    def _format_parity_data(self):
        """
        Format parity data to account for degeneracies.

        """
        trim_labels = [key for key in self.trim_data["up"].keys()]

        nele = self.v2t_output["up"].num_occ_bands
        trim_parities_formatted = {}
        trim_energies_formatted = {}

        for label in trim_labels:
            trim_parities_formatted[label] = np.zeros(int(nele/2))
            trim_energies_formatted[label] = np.zeros(int(nele/2))
            count = 0


            if np.any([int(i) == 1 for i in self.trim_data["up"][label]["iden"][:]]):
                raise RuntimeError('Vasp2trace does not show completely degenrate bands at %s.' % label)

            iden_sum = int(np.sum(self.trim_data["up"][label]["iden"][:]))
            if nele < iden_sum and \
               int(self.trim_data["up"][label]["parity"][-1]) == 0:
                raise RuntimeError(
                    'Cannot tell the parity of the highest occupied state at %s.' % label)

            for i in range(len(self.trim_data["up"][label]["energies"])):
                iden = int(self.trim_data["up"][label]["iden"][i])
                parity = int(self.trim_data["up"][label]["parity"][i])
                energy = self.trim_data["up"][label]["energies"][i]

                if iden == 2:
                    trim_parities_formatted[label][count] = parity/abs(parity)
                    trim_energies_formatted[label][count] = energy
                    count += 1

                elif iden > 2:
                    for j in range(int(abs(iden)/2)):
                        if abs(parity) > 1.0:
                            trim_parities_formatted[label][count] = parity / \
                                abs(parity)
                            trim_energies_formatted[label][count] = energy

                        else:  # - Make zeros from four-fold degenerate states equal to -1
                            if j == 0:
                                trim_parities_formatted[label][count] = -1
                                trim_energies_formatted[label][count] = energy
                            else:
                                trim_parities_formatted[label][count] = 1
                                trim_energies_formatted[label][count] = energy

                        count += 1
                        if count == nele/2:
                            break

        return trim_parities_formatted, trim_energies_formatted

    @staticmethod
    def _get_band_subspace(tol=2, trim_energies_formatted=None):
        """
        Find a subgroup of valence bands for topology analysis that are gapped from lower lying bands topologically trivial bands.

        Args:
            tol (float): Tolerance for average energy difference between bands at TRIM points to define independent band group.

        """

        points = [key for key in trim_energies_formatted.keys()]
        delta_e = {}

        mark = None

        band_energies = trim_energies_formatted[points[0]]
        nbands = len(trim_energies_formatted[points[0]])

        if tol == -1:
            return (nbands+1)
        else:
            points = [key for key in trim_energies_formatted.keys()]
            delta_e = {}

            mark = None

            band_energies = trim_energies_formatted[points[0]]
            nbands = len(trim_energies_formatted[points[0]])

            for band_num in reversed(range(nbands - 1)):
                diff = abs(band_energies[band_num + 1] -
                           band_energies[band_num])

                if diff >= tol:
                    for point in points:
                        band_energies2 = trim_energies_formatted[point]

                        diff2 = abs(
                            band_energies2[band_num + 1] - band_energies2[band_num])

                        if diff2 < tol:
                            break
                        else:
                            if point == points[-1]:
                                mark = band_num

                if mark is not None:
                    break

            return (nbands-mark)

    @staticmethod
    def screen_semimetal(trim_parities):
        """
        Parity criteria screening for metallic band structures to predict if nonmagnetic Weyl semimetal phase is allowed.

        Args:
            trim_parities (dict): non-spin-polarized trim parities.

        Returns:
            semimetal (int): -1 (system MIGHT be a semimetal) or 1 (not a semimetal).

        """

        # Count total number of odd parity states over all TRIMs
        num_odd_states = 0

        for trim_label, band_parities in trim_parities["up"].items():
            num_odd_at_trim = np.sum(
                np.fromiter((1 for i in band_parities if i < 0), dtype=int)
            )

            num_odd_states += num_odd_at_trim

        if num_odd_states % 2 == 1:
            semimetal = -1
        else:
            semimetal = 1

        return semimetal

    @staticmethod
    def screen_magnetic_parity(trim_parities):
        """
        Screen candidate inversion-symmetric magnetic topological materials from band parity criteria.

        Returns a dictionary of *allowed* magnetic topological properties where their bool values indicate if the property is allowed.

        Requires a spin-polarized VASP calculation and trace_up and trace_dn from vasp2trace v2.

        REF: Turner et al., PRB 85, 165120 (2012).

        Args:
            trim_parities (dict): 'up' and 'down' spin channel occupied band parities at TRIM points.

        Returns:
            mag_screen (dict): Magnetic topological properties from band parities.

        """

        mag_screen = {
            "insulator": False,
            "semimetal_candidate": False,
            "polarization_bqhc": False,
            "magnetoelectric": False,
        }

        # Count total number of odd parity states over all TRIMs
        num_odd_states = 0

        # Check if any individual TRIM pt has an odd num of odd states
        odd_total_at_trim = False

        for spin in ["up", "down"]:
            for trim_label, band_parities in trim_parities[spin].items():
                num_odd_at_trim = np.sum(
                    np.fromiter((1 for i in band_parities if i < 0), dtype=int)
                )

                num_odd_states += num_odd_at_trim

                if num_odd_at_trim % 2 == 1:
                    odd_total_at_trim = True

        # Odd number of odd states -> CAN'T be an insulator
        # Might be a Weyl semimetal
        if num_odd_states % 2 == 1:
            mag_screen["insulator"] = False
            mag_screen["semimetal_candidate"] = True
            mag_screen["polarization_bqhc"] = False
            mag_screen["magnetoelectric"] = False
        else:
            mag_screen["insulator"] = True

        # Further screening for magnetic insulators
        if mag_screen["insulator"]:

            # Check if any individual TRIM pt has an odd num of odd states
            # Either electrostatic polarization OR bulk quantized Hall
            # conductivity
            if odd_total_at_trim:
                mag_screen["polarization_bqhc"] = True

            # If no BQHC
            # num_odd_states = 2*k for any odd integer k
            k = num_odd_states / 2
            if k.is_integer():
                if int(k) % 2 == 1:  # odd
                    mag_screen["magnetoelectric"] = True

        return mag_screen


class StructureDimensionality(MSONable):
    def __init__(
        self,
        structure,
        structure_graph=None,
        larsen_dim=None,
        cheon_dim=None,
        gorai_dim=None,
    ):
        """
        This class uses 3 algorithms implemented in pymatgen to automate recognition of low-dimensional materials.

        Args:
            structure (object): pmg Structure object.

            structure_graph (object): pmg StructureGraph object.

            larsen_dim (int): As defined in P. M. Larsen et al., Phys. Rev. Materials 3, 034003 (2019).

            cheon_dim (int): As defined in Cheon, G. et al., Nano Lett. 2017.

            gorai_dim (int): As defined in Gorai, P. et al., J. Mater. Chem. A 2, 4136 (2016).

        """

        self.structure = structure
        self.structure_graph = structure_graph
        self.larsen_dim = larsen_dim
        self.cheon_dim = cheon_dim
        self.gorai_dim = gorai_dim

        # Default to MinimumDistanceNN for generating structure graph.
        sgraph = MinimumDistanceNN().get_bonded_structure(structure)

        self.structure_graph = sgraph

        # Get Larsen dimensionality
        self.larsen_dim = get_dimensionality_larsen(self.structure_graph)

    def get_cheon_gorai_dim(self):
        """Convenience method for getting Cheon and Gorai dims. These algorithms take some time.

        Returns:
            None: (sets instance variables).

        """
        cheon_dim_str = get_dimensionality_cheon(self.structure)

        if cheon_dim_str == "0D":
            cheon_dim = 0
        elif cheon_dim_str == "1D":
            cheon_dim = 1
        elif cheon_dim_str == "2D":
            cheon_dim = 2
        elif cheon_dim_str == "3D":
            cheon_dim = 3
        else:
            cheon_dim = None

        self.cheon_dim = cheon_dim

        self.gorai_dim = get_dimensionality_gorai(self.structure)
