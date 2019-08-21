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

    def compute_z2(self):
        """
        Compute Z2 = (v0, v1v2v3) strong and weak topological indices in 3D from TRIM band parities.

        """

        pass

    def _get_band_subspace(self):
        """
        Find a subgroup of valence bands for topology analysis that are gapped from lower lying bands topologically trivial bands.

        """

        pass
