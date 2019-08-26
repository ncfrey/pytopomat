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

        # Spin-polarized
        elif path.isfile("trace_up.txt") and path.isfile("trace_dn.txt"):
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

        with open(vasp2trace_output, "r") as file:
            lines = file.readlines()

            # Get header info
            num_occ_bands = int(lines[0])
            soc = int(lines[1])  # No: 0, Yes: 1
            num_symm_ops = int(lines[2])
            symm_ops = np.loadtxt(lines[3 : 3 + num_symm_ops])
            num_max_kvec = int(lines[3 + num_symm_ops])
            kvecs = np.loadtxt(
                lines[4 + num_symm_ops : 4 + num_symm_ops + num_max_kvec]
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
                num_kvec_symm_ops[idx] = int(lines[start_block])
                soilcg = [
                    int(i.strip("\n"))
                    for i in lines[start_block + 1].split(" ")
                    if i.strip("\n")
                ]
                symm_ops_in_little_cogroup[idx] = soilcg

                trace = np.loadtxt(trace_str)
                traces[idx] = trace

        # Set instance attributes
        self.num_occ_bands = num_occ_bands
        self.soc = soc
        self.num_symm_ops = num_symm_ops
        self.symm_ops = symm_ops
        self.num_max_kvec = num_max_kvec
        self.kvecs = kvecs
        self.num_kvec_symm_ops = num_kvec_symm_ops
        self.symm_ops_in_little_cogroup = symm_ops_in_little_cogroup
        self.traces = traces


class BandParity(MSONable):
    def __init__(self, v2t_output=None, trim_parities=None, spin_polarized=None):
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

            trim_parities (dict): Maps TRIM point labels to band parity eigenvals.

            spin_polarized (bool): Spin-polarized or not.

        TODO:
            * Try to find a gapped subspace of Bloch bands
            * Compute overall parity and Z2=(v0, v1v2v3)
            * Report spin-polarized parity filters
        """

        self.v2t_output = v2t_output
        self.trim_parities = trim_parities
        self.spin_polarized = spin_polarized

        # Check if spin-polarized or not
        if "down" in v2t_output.keys():  # spin polarized
            self.spin_polarized = True
        else:
            self.spin_polarized = False

        if self.spin_polarized:
            self.trim_parities = {}

            # Up spin
            parity_op_index_up = self._get_parity_op(self.v2t_output["up"].symm_ops)
            self.trim_parities["up"] = self.get_trim_parities(
                parity_op_index_up, self.v2t_output["up"]
            )

            # Down spin
            parity_op_index_dn = self._get_parity_op(self.v2t_output["down"].symm_ops)
            self.trim_parities["down"] = self.get_trim_parities(
                parity_op_index_dn, self.v2t_output["down"]
            )

        else:
            self.trim_parities = {}
            parity_op_index = self._get_parity_op(self.v2t_output["up"].symm_ops)
            self.trim_parities["up"] = self.get_trim_parities(
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

        parity_mat = np.array([-1, 0, 0, 0, -1, 0, 0, 0, -1])  # x,y,z -> -x,-y,-z

        for idx, symm_op in enumerate(symm_ops):
            rot_mat = symm_op[0:9]  # Rotation matrix
            trans_mat = symm_op[9:12]  # Translation matrix

            # Find the parity operator
            if np.array_equal(rot_mat, parity_mat):
                parity_op_index = idx + 1  # SymmOps are 1 indexed

        return parity_op_index

    @staticmethod
    def get_trim_parities(parity_op_index, v2t_output):
        """Tabulate parity eigenvals for all occupied bands over all TRIM points.

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

        trim_parities = {trim_label: [] for trim_label in trim_labels}

        for idx, kvec in enumerate(v2to.kvecs):
            for trim_pt, trim_label in trims.items():
                if np.array_equal(kvec, trim_pt):  # Check if a TRIM

                    # Index of parity op for this kvec
                    kvec_parity_idx = v2to.symm_ops_in_little_cogroup[idx].index(
                        parity_op_index
                    )

                    kvec_traces = v2to.traces[idx]
                    for band_index, band in enumerate(kvec_traces):

                        # Real part of parity eigenval
                        band_parity_eigenval = band[3 + 2 * kvec_parity_idx]

                        trim_parities[trim_label].append(band_parity_eigenval)

        return trim_parities

    def compute_z2(self, tol=0.2):
        """
        Compute Z2 topological indices (index) in 3D (2D) from TRIM band parities.

        Args:
            tol (float): Tolerance for average energy difference between bands at TRIM points to define independent band group.

        """

        trim_labels = [key for key in self.trim_parities["up"].keys()]

        iband = self._get_band_subspace(tol)

        if len(trim_labels) == 8:
            Z2 = np.ones(4, dtype=int)


            for label in trim_labels:
                delta = 1
                for parity in self.trim_parities["up"][label][iband:]:
                    delta *= parity/abs(parity)
                                       
                Z2[0] *= delta                  

                if label in ["x", "s", "u", "r"]:
                    Z2[1] *= delta
                if label in ["y", "s", "t", "r"]:
                    Z2[2] *= delta
                if label in ["z", "t", "u", "r"]:
                    Z2[3] *= delta
 

            return ((Z2-1)/-2)+0    

        elif len(trim_labels) == 4:
            Z2 = np.ones(1, dtype=int)

            for label in trim_labels:
                delta = 1
                for parity in self.trim_parities["up"][label][iband:]:
                    delta *= parity/abs(parity)

                Z2 *= delta

            return ((Z2-1)/-2)+0

        else:
            raise RuntimeError("Incorrect number of k-points in vasp2trace output.")


    def _get_band_subspace(self, tol=0.2):
        """
        Find a subgroup of valence bands for topology analysis that are gapped from lower lying bands topologically trivial bands.

        Args:
            tol (float): Tolerance for average energy difference between bands at TRIM points to define independent band group.

        """

        points = [num for num in self.v2t_output['up'].traces.keys()]
        
        for point in points:
            band_data = self.v2t_output['up'].traces[point]
            nbands = len(self.v2t_output['up'].traces[point])

            delta_e = np.zeros(nbands)

            for band_num in range(nbands-1):
                diff = abs(band_data[band_num+1][2] - band_data[band_num][2])
                delta_e[band_num] += diff

        delta_e = delta_e/len(points)

        max_diff = delta_e[0]
        for ind in range(nbands):
            if delta_e[ind]-max_diff >= tol:
                max_diff = delta_e[ind]

        return np.argwhere(delta_e==max_diff)[0][0]+1

    @staticmethod
    def screen_semimetal(trim_parities):
        """
        Parity criteria screening for metallic band structures to predict if Weyl semimetal phase is allowed.

        Args:
            trim_parities (dict): non-spin-polarized trim parities.

        Returns:
            semimetal (int): -1 (system MIGHT be a semimetal) or 1 (not a semimetal).

        """

        # Count total number of odd parity states over all TRIMs
        num_odd_states = 0

        for trim_label, band_parities in trim_parities['up'].items():
            num_odd_at_trim = np.sum(np.fromiter((1 for i in band_parities if i < 0), dtype=int))

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

        REF: Turner et al., PRB 85, 165120 (2012).

        Args:
            trim_parities (dict): 'up' and 'down' spin channel occupied band parities at TRIM points.

        Returns:
            mag_screen (dict): Magnetic topological properties from band parities.

        """

        mag_screen = {"insulator": False, "polarization_bqhc": False, "magnetoelectric": False}

        # Count total number of odd parity states over all TRIMs
        num_odd_states = 0

        # Check if any individual TRIM pt has an odd num of odd states
        odd_total_at_trim = False

        for spin in ['up', 'down']:
            for trim_label, band_parities in trim_parities[spin].items():
                num_odd_at_trim = np.sum(np.fromiter((1 for i in band_parities if i < 0), dtype=int))

                num_odd_states += num_odd_at_trim

                if num_odd_at_trim % 2 == 1:
                    odd_total_at_trim = True

        # Odd number of odd states -> CAN'T be an insulator
        # Might be a Weyl semimetal
        if num_odd_states % 2 == 1:
            mag_screen["insulator"] = False
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













