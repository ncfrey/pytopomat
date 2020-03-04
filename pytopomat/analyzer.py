"""
Compute topological invariants.

This module offers a high level framework for analyzing topological materials in a 
high-throughput context with VASP, Z2Pack, irvsp, and Vasp2Trace.

"""

import numpy as np

import warnings

from monty.json import MSONable

from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.analysis.dimensionality import (
    get_dimensionality_larsen,
    get_dimensionality_cheon,
    get_dimensionality_gorai,
)

__author__ = "Nathan C. Frey, Jason Munro"
__copyright__ = "MIT License"
__version__ = "0.0.1"
__maintainer__ = "Nathan C. Frey, Jason, Munro"
__email__ = "ncfrey@lbl.gov, jmunro@lbl.gov"
__status__ = "Development"
__date__ = "August 2019"


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

        Todo:
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
            parity_op_index_up = self._get_parity_op(self.v2t_output["up"].symm_ops)
            self.trim_data["up"] = self.get_trim_data(
                parity_op_index_up, self.v2t_output["up"]
            )

            # Down spin
            parity_op_index_dn = self._get_parity_op(self.v2t_output["down"].symm_ops)
            self.trim_data["down"] = self.get_trim_data(
                parity_op_index_dn, self.v2t_output["down"]
            )

        else:
            self.trim_data = {}

            parity_op_index = self._get_parity_op(self.v2t_output["up"].symm_ops)
            self.trim_data["up"] = self.get_trim_data(
                parity_op_index, self.v2t_output["up"]
            )

    @staticmethod
    def _get_parity_op(symm_ops):
        """
        Find parity in the list of SymmOps.

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
                raise RuntimeError(
                    "No non-trivial symmetry operations in vasp2trace output!"
                )

            # Find the parity operator
            if np.array_equal(rot_mat, parity_mat):
                parity_op_index = idx + 1  # SymmOps are 1 indexed

        if parity_op_index == None:
            raise RuntimeError("Parity operation not found in vasp2trace output!")
        else:
            return parity_op_index

    @staticmethod
    def get_trim_data(parity_op_index, v2t_output):
        """
        Tabulate parity and identity eigenvals, as well as energies for all 
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

        trim_data = {
            trim_label: {"energies": [], "iden": [], "parity": []}
            for trim_label in trim_labels
        }

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

                        trim_data[trim_label]["parity"].append(band_parity_eigenval)
                        trim_data[trim_label]["iden"].append(band_iden_eigenval)
                        trim_data[trim_label]["energies"].append(band_energy)

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

        trim_parities_set, trim_energies_set = self._format_parity_data()
        trim_parities = trim_parities_set["up"]
        trim_energies = trim_energies_set["up"]

        iband = self._get_band_subspace(tol=tol, trim_energies_formatted=trim_energies)
        print("Only considering last %i pairs of bands." % (iband - 1))

        if len(trim_labels) == 8:
            Z2 = np.ones(4, dtype=int)

            for label in trim_labels:
                delta = 1
                for parity in trim_parities[label][: (-1 * iband) : -1]:
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
                for parity in trim_parities[label][: (-1 * iband) : -1]:
                    delta *= parity

                Z2 *= delta

            return ((Z2 - 1) / -2) + 0

        else:
            raise RuntimeError("Incorrect number of k-points in vasp2trace output.")

    def _format_parity_data(self):
        """
        Format parity data to account for degeneracies.

        """

        spin_polarized = self.spin_polarized

        trim_labels = [key for key in self.trim_data["up"].keys()]
        nele = self.v2t_output["up"].num_occ_bands

        if not spin_polarized:
            spins = ["up"]
            criteria = nele / 2
        else:
            spins = ["up", "down"]
            criteria = nele

        trim_parities_formatted = {spin: {} for spin in spins}
        trim_energies_formatted = {spin: {} for spin in spins}

        for label in trim_labels:
            for spin in spins:
                trim_parities_formatted[spin][label] = np.zeros(int(criteria))
                trim_energies_formatted[spin][label] = np.zeros(int(criteria))
                count = 0

                if not spin_polarized:
                    if np.any(
                        [int(i) == 1 for i in self.trim_data[spin][label]["iden"][:]]
                    ):
                        raise RuntimeError(
                            "Non-spin polarized selected, but trace data does not show doubly degenerate bands at %s."
                            % label
                        )

                    iden_sum = int(np.sum(self.trim_data[spin][label]["iden"][:]))
                    if (
                        nele < iden_sum
                        and int(self.trim_data["up"][label]["parity"][-1]) == 0
                    ):
                        raise RuntimeError(
                            "Cannot tell the parity of the highest occupied state at %s."
                            % label
                        )
                else:
                    if np.all(
                        [int(i) != 1.0 for i in self.trim_data[spin][label]["iden"][:]]
                    ):
                        warnings.warn(
                            "Spin polarized selected, but at least one TRIM point shows all doubly degenerate bands."
                        )

                    iden_sum = int(np.sum(self.trim_data[spin][label]["iden"][:]))
                    if (
                        nele < iden_sum
                        and int(self.trim_data[spin][label]["parity"][-1]) > 1
                    ):
                        raise RuntimeError(
                            "Cannot tell the parity of the highest occupied state at %s."
                            % label
                        )

                for i in range(len(self.trim_data[spin][label]["energies"])):
                    iden = int(self.trim_data[spin][label]["iden"][i])
                    parity = int(self.trim_data[spin][label]["parity"][i])
                    energy = self.trim_data[spin][label]["energies"][i]

                    if iden == 2:
                        if not spin_polarized:
                            trim_parities_formatted[spin][label][count] = parity / abs(
                                parity
                            )
                            trim_energies_formatted[spin][label][count] = energy
                            count += 1
                        else:
                            for j in range(0, 2):
                                if parity != 0:
                                    trim_parities_formatted[spin][label][
                                        count
                                    ] = parity / abs(parity)
                                    trim_energies_formatted[spin][label][count] = energy

                                else:  # - Make zeros from two-fold degenerate states equal to -1
                                    if j == 0:
                                        trim_parities_formatted[spin][label][count] = -1
                                        trim_energies_formatted[spin][label][
                                            count
                                        ] = energy
                                    else:
                                        trim_parities_formatted[spin][label][count] = 1
                                        trim_energies_formatted[spin][label][
                                            count
                                        ] = energy

                                count += 1
                                if count == criteria:
                                    break

                    elif iden > 2:
                        for j in range(int(abs(iden) / 2)):
                            if abs(parity) > 1.0:
                                trim_parities_formatted[spin][label][
                                    count
                                ] = parity / abs(parity)
                                trim_energies_formatted[spin][label][count] = energy

                            else:  # - Make zeros from four-fold degenerate states equal to -1
                                if j == 0:
                                    trim_parities_formatted[spin][label][count] = -1
                                    trim_energies_formatted[spin][label][count] = energy
                                else:
                                    trim_parities_formatted[spin][label][count] = 1
                                    trim_energies_formatted[spin][label][count] = energy

                            count += 1
                            if count == criteria:
                                break

                    elif iden == 1:
                        trim_parities_formatted[spin][label][count] = parity
                        trim_energies_formatted[spin][label][count] = energy

                        count += 1

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
            return nbands + 1
        else:
            points = [key for key in trim_energies_formatted.keys()]
            delta_e = {}

            mark = None

            band_energies = trim_energies_formatted[points[0]]
            nbands = len(trim_energies_formatted[points[0]])

            for band_num in reversed(range(nbands - 1)):
                diff = abs(band_energies[band_num + 1] - band_energies[band_num])

                if diff >= tol:
                    for point in points:
                        band_energies2 = trim_energies_formatted[point]

                        diff2 = abs(
                            band_energies2[band_num + 1] - band_energies2[band_num]
                        )

                        if diff2 < tol:
                            break
                        else:
                            if point == points[-1]:
                                mark = band_num

                if mark is not None:
                    break

            return nbands - mark

    def screen_semimetal(self):
        """
        Parity criteria screening for metallic band structures to predict if nonmagnetic Weyl semimetal phase is allowed.

        Returns:
            semimetal (int): -1 (system MIGHT be a semimetal) or 1 (not a semimetal).

        """

        trim_parities, trim_energies = self._format_parity_data()

        # Count total number of odd parity states over all TRIMs
        num_odd_states = 0

        for trim_label, parities in trim_parities["up"].items():
            num_odd_at_trim = np.sum(
                np.fromiter((1 for i in parities if i < 0), dtype=int)
            )

            num_odd_states += num_odd_at_trim

        if num_odd_states % 2 == 1:
            semimetal = -1
        else:
            semimetal = 1

        return semimetal

    def screen_magnetic_parity(self):
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

        trim_parities, trim_energies = self._format_parity_data()

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
