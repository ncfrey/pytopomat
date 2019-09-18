import z2pack
from monty.json import MSONable

"""
This module offers a high level framework for analyzing topological materials in a high-throughput context with VASP and Z2Pack.
"""

__author__ = "Nathan C. Frey, Jason Munro"
__copyright__ = "MIT License"
__version__ = "0.0.1"
__maintainer__ = "Nathan C. Frey, Jason Munro"
__email__ = "ncfrey@lbl.gov, jmunro@lbl.gov"
__status__ = "Development"
__date__ = "August 2019"


class Z2PackCaller:
    def __init__(
        self, input_dir="input", surface="kz_0", vasp_cmd="srun vasp_ncl >& log"
    ):
        """A class for analyzing band structure topology and diagnosing non-trivial topological phases.

        Create a z2pack.fp.System instance for vasp and Wannier90 that points to inputs and allows for dynamic calling of vasp.

        When called from a root directory, input files (POSCAR, INCAR, etc.) must be in a folder called 'input'.

        Examples of possible Brillouin zone surfaces:
        [0, t1 / 2, t2]  : k_x = 0
        [1/2, t1 /2 , t2]  : k_x = 1/2

        Required VASP flags:
            LWANNIER90 = .TRUE.
            LWRITE_MMN_AMN = .TRUE.
            ISYM = -1
            NBANDS = (set to divisible by num of cores so no extra bands are added)

        Required Wannier90 flags:
            exclude_bands = (set to exclude unoccupied bands)

        Args:
            input_dir (str): Path to input vasp and Wannier90 input files.
            surface (function): Brillouin zone surface, defaults to (kx, ky, 0).
            vasp_cmd (str): Specify command to run vasp that the class should use. E.g. "mpirun vasp_ncl >& log".

        Parameters:
            system (z2pack System object): Configuration for dynamically calling vasp within z2pack.
            surface (str): Labels the surface, e.g. [t1/2, t2, 0] <-> "kz_0".

        This module makes extensive use of the z2pack tool for calculating topological invariants to identify topological phases. It is mainly meant to be used in band structure workflows for high throughput classification of band topology.

        If you use this module, please cite the following papers:

        Dominik Gresch, Gabriel Autès, Oleg V. Yazyev, Matthias Troyer, David Vanderbilt, B. Andrei Bernevig, and Alexey A. Soluyanov “Z2Pack: Numerical Implementation of Hybrid Wannier Centers for Identifying Topological Materials” [PhysRevB.95.075146]

        Alexey A. Soluyanov and David Vanderbilt “Computing topological invariants without inversion symmetry” [PhysRevB.83.235401]

        """

        # Surface label -> lambda function parameterization
        self.surface = surface
        self.input_dir = input_dir

        # Define input file locations
        input_files = ["CHGCAR", "INCAR", "POSCAR", "POTCAR", "wannier90.win"]
        input_files = [self.input_dir + "/" + s for s in input_files]

        # Create k-point inputs for VASP
        kpt_fct = z2pack.fp.kpoint.vasp

        system = z2pack.fp.System(
            input_files=input_files,
            kpt_fct=kpt_fct,
            kpt_path="KPOINTS",
            command=vasp_cmd,
            mmn_path="wannier90.mmn",
        )

        self.system = system

    def run(self, z2_settings=None):
        """Calculate Wannier charge centers on the BZ surface.

        Args:
            z2_settings (dict): Optional settings for specifying convergence criteria. Check z2_defaults for keywords.

        """

        # z2 calculation defaults
        z2d = {
            "pos_tol": 0.01,  # change in Wannier charge center pos
            "gap_tol": 0.3,  # Limit for closeness of lines on surface
            "move_tol": 0.3,  # Movement of WCC between neighbor lines
            "num_lines": 11,  # Min num of lines to calculate
            "min_neighbour_dist": 0.01,  # Min dist between lines
            "iterator": range(8, 27, 2),  # Num of kpts to iterate over
            "load": True,  # Start from most recent calc
            "save_file": self.surface + "_z2run.json",  # Serialize results
        }

        # User defined setting updates to defaults
        if z2_settings:
            for k, v in z2_settings.items():
                d = {k: v}
                z2d.update(d)

        # Create a Brillouin zone surface for calculating the Wilson loop / Wannier charge centers (defaults to k_z = 0 surface)
        surfaces = {
            "kx_0": lambda s, t: [0, s / 2, t],
            "kx_1": lambda s, t: [0.5, s / 2, t],
            "ky_0": lambda s, t: [s / 2, 0, t],
            "ky_1": lambda s, t: [s / 2, 0.5, t],
            "kz_0": lambda s, t: [s / 2, t, 0],
            "kz_1": lambda s, t: [s / 2, t, 0.5],
        }

        # Calculate WCC on the Brillouin zone surface.
        result = z2pack.surface.run(
            system=self.system,
            surface=surfaces[self.surface],
            pos_tol=z2d["pos_tol"],
            gap_tol=z2d["gap_tol"],
            move_tol=z2d["move_tol"],
            num_lines=z2d["num_lines"],
            min_neighbour_dist=z2d["min_neighbour_dist"],
            iterator=z2d["iterator"],
            load=z2d["load"],
            save_file=z2d["save_file"],
        )

        self.output = Z2Output(result, self.surface)


class Z2Output(MSONable):
    def __init__(self, result, surface, chern_number=None, z2_invariant=None):
        """
        Class for storing results of band topology analysis.

        Args:
            result (object): Output from z2pack.surface.run()
            surface (str): TRI BZ surface label.
            chern_number (int): Chern number.
            z2_invariant (int): Z2 invariant. 
            
        """

        self._result = result
        self.surface = surface
        self.chern_number = chern_number
        self.z2_invariant = z2_invariant

        self._parse_result(result)

    def _parse_result(self, result):

        # Topological invariants
        chern_number = z2pack.invariant.chern(result)
        z2_invariant = z2pack.invariant.z2(result)

        self.chern_number = chern_number
        self.z2_invariant = z2_invariant
