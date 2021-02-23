"""
Wflows using irvsp, vasp2trace, and Z2Pack.

"""

import numpy as np
import itertools
import os

from uuid import uuid4

from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.sets import MPStaticSet, MPRelaxSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from atomate.vasp.config import STABILITY_CHECK, VASP_CMD, DB_FILE, ADD_WF_METADATA
from atomate.vasp.powerups import (
    add_stability_check,
    add_modify_incar,
    add_wf_metadata,
    add_common_powerups,
    add_additional_fields_to_taskdocs,
    add_tags,
)
from atomate.vasp.workflows.base.core import get_wf
from atomate.vasp.fireworks.core import OptimizeFW, StaticFW

from fireworks import Workflow

from pytopomat.analyzer import StructureDimensionality
from pytopomat.workflows.fireworks import Z2PackFW, InvariantFW


__author__ = "Jason Munro, Nathan C. Frey"
__copyright__ = "MIT License"
__version__ = "0.0.1"
__maintainer__ = "Jason Munro, Nathan C. Frey"
__email__ = "jmunro@lbl.gov, ncfrey@lbl.gov"
__status__ = "Development"
__date__ = "August 2019"


def wf_irrep(structure, magnetic=False, soc=False, c=None):
    """
    Fireworks workflow for running an irrep calculation.

    Args:
        structure (Structure): Pymatgen structure object
        magnetic (bool): Whether the calculation is on a magnetic structure
        soc (bool): Spin-orbit coupling included

    Returns:
        Workflow

    """

    c = c or {}
    vasp_cmd = c.get("VASP_CMD", VASP_CMD)
    db_file = c.get("DB_FILE", DB_FILE)

    uuid = str(uuid4())
    wf_meta = {"wf_uuid": uuid, "wf_name": "Irrep WF"}

    magmoms = None

    if magnetic and "magmom" in structure.site_properties:
        magmoms_orig = structure.site_properties["magmom"]
        magmoms = [str(m) for m in magmoms_orig]
        magmoms = " ".join(magmoms)
    elif magnetic:
        raise RuntimeError(
            "Structure must have magnetic moments in site_properties for magnetic calcualtion!"
        )

    ncoords = 3 * len(structure.sites)

    nbands = 0

    for site in structure.sites:
        nbands += site.species.total_electrons

    trim_kpoints = Kpoints(
        comment="TRIM Points",
        num_kpts=8,
        style=Kpoints.supported_modes.Reciprocal,
        kpts=(
            (0, 0, 0),
            (0.5, 0, 0),
            (0, 0.5, 0),
            (0, 0, 0.5),
            (0.5, 0.5, 0),
            (0, 0.5, 0.5),
            (0.5, 0, 0.5),
            (0.5, 0.5, 0.5),
        ),
        kpts_shift=(0, 0, 0),
        kpts_weights=[1, 1, 1, 1, 1, 1, 1, 1],
        coord_type="Reciprocal",
        labels=["gamma", "x", "y", "z", "s", "t", "u", "r"],
        tet_number=0,
        tet_weight=0,
        tet_connections=None,
    )

    # params dicts for wf
    params = [
        {},  # optimization
        {},  # standardization
        {},  # static
        {
            "input_set_overrides": {
                "other_params": {"user_kpoints_settings": trim_kpoints}
            }
        },  # nscf
        {"wf_uuid": uuid},  # irrep
    ]

    yaml_spec = "irvsp.yaml"

    wf = get_wf(
        structure,
        yaml_spec,
        params=params,
        vis=MPStaticSet(structure, potcar_functional="PBE_54", force_gamma=True),
        common_params={"vasp_cmd": vasp_cmd, "db_file": db_file},
        wf_metadata=wf_meta,
    )

    dim_data = StructureDimensionality(structure)

    if np.any(
        [
            dim == 2
            for dim in [dim_data.larsen_dim, dim_data.cheon_dim, dim_data.gorai_dim]
        ]
    ):
        wf = add_modify_incar(
            wf,
            modify_incar_params={
                "incar_update": {"IVDW": 11, "EDIFFG": 0.005, "IBRION": 2, "NSW": 100}
            },
            fw_name_constraint="structure optimization",
        )
    else:
        wf = add_modify_incar(
            wf,
            modify_incar_params={
                "incar_update": {"EDIFFG": 0.005, "IBRION": 2, "NSW": 100}
            },
            fw_name_constraint="structure optimization",
        )

    wf = add_modify_incar(
        wf,
        modify_incar_params={
            "incar_update": {"ADDGRID": ".TRUE.", "LASPH": ".TRUE.", "GGA": "PS"}
        },
    )

    # Includ ncl magmoms with saxis = (0, 0, 1)
    if magnetic and soc:
        magmoms = []
        for m in magmoms_orig:
            magmoms += [0.0, 0.0, m]
        magmoms = [str(m) for m in magmoms]
        magmoms = " ".join(magmoms)

        wf = add_modify_incar(
            wf,
            modify_incar_params={"incar_update": {"ISYM": 2, "MAGMOM": "%s" % magmoms}},
        )

    if magnetic and not soc:
        # Include magmoms in every calculation
        wf = add_modify_incar(
            wf,
            modify_incar_params={
                "incar_update": {"ISYM": 2, "MAGMOM": "%s" % magmoms, "ISPIN": 2}
            },
        )

    wf = add_modify_incar(
        wf,
        modify_incar_params={
            "incar_update": {
                "ISYM": 2,
                "LSORBIT": ".TRUE." if soc else ".FALSE.",
                "MAGMOM": "%s" % magmoms if magnetic else "%i*0.0" % ncoords,
                "ISPIN": 2 if magnetic and not soc else 1,
                "LWAVE": ".TRUE.",
                # "NBANDS": nbands,
            }
        },
        fw_name_constraint="nscf",
    )

    wf = add_common_powerups(wf, c)

    wf = add_additional_fields_to_taskdocs(
        wf, {"wf_meta": wf_meta}, task_name_constraint="VaspToDb"
    )

    if c.get("STABILITY_CHECK", STABILITY_CHECK):
        wf = add_stability_check(wf, fw_name_constraint="structure optimization")

    if c.get("ADD_WF_METADATA", ADD_WF_METADATA):
        wf = add_wf_metadata(wf, structure)

    return wf


def wf_irvsp(structure, magnetic=False, soc=False, v2t=False, c=None):
    """
    Fireworks workflow for running an irvsp calculation.
    Optionally performs a vasp2trace calculation.

    Args:
        structure (Structure): Pymatgen structure object
        magnetic (bool): Whether the calculation is on a magnetic structure
        soc (bool): Spin-orbit coupling included
        v2t (bool): Do a vasp2trace calculation in addition to irvsp.

    Returns:
        Workflow

    """

    c = c or {}
    vasp_cmd = c.get("VASP_CMD", VASP_CMD)
    db_file = c.get("DB_FILE", DB_FILE)

    uuid = str(uuid4())
    wf_meta = {"wf_uuid": uuid, "wf_name": "Irvsp WF"}

    magmoms = None

    if magnetic and "magmom" in structure.site_properties:
        magmoms_orig = structure.site_properties["magmom"]
        magmoms = [str(m) for m in magmoms_orig]
        magmoms = " ".join(magmoms)
    elif magnetic:
        raise RuntimeError(
            "Structure must have magnetic moments in site_properties for magnetic calcualtion!"
        )

    ncoords = 3 * len(structure.sites)

    nbands = 0

    for site in structure.sites:
        nbands += site.species.total_electrons

    trim_kpoints = Kpoints(
        comment="TRIM Points",
        num_kpts=8,
        style=Kpoints.supported_modes.Reciprocal,
        kpts=(
            (0, 0, 0),
            (0.5, 0, 0),
            (0, 0.5, 0),
            (0, 0, 0.5),
            (0.5, 0.5, 0),
            (0, 0.5, 0.5),
            (0.5, 0, 0.5),
            (0.5, 0.5, 0.5),
        ),
        kpts_shift=(0, 0, 0),
        kpts_weights=[1, 1, 1, 1, 1, 1, 1, 1],
        coord_type="Reciprocal",
        labels=["gamma", "x", "y", "z", "s", "t", "u", "r"],
        tet_number=0,
        tet_weight=0,
        tet_connections=None,
    )

    # params dicts for wf
    params = [
        {},  # optimization
        {},  # standardization
        {},  # static
        {
            "input_set_overrides": {
                "other_params": {"user_kpoints_settings": trim_kpoints}
            }
        },  # nscf
        {"wf_uuid": uuid},  # irvsp
    ]

    if magnetic and v2t:
        yaml_spec = "irvsp_v2t_magnetic.yaml"
        params.append({})
    elif v2t:
        yaml_spec = "irvsp_v2t.yaml"
        params.append({})
    else:
        yaml_spec = "irvsp.yaml"

    wf = get_wf(
        structure,
        yaml_spec,
        params=params,
        vis=MPStaticSet(structure, potcar_functional="PBE_54", force_gamma=True),
        common_params={"vasp_cmd": vasp_cmd, "db_file": db_file},
        wf_metadata=wf_meta,
    )

    dim_data = StructureDimensionality(structure)

    if np.any(
        [
            dim == 2
            for dim in [dim_data.larsen_dim, dim_data.cheon_dim, dim_data.gorai_dim]
        ]
    ):
        wf = add_modify_incar(
            wf,
            modify_incar_params={
                "incar_update": {"IVDW": 11, "EDIFFG": 0.005, "IBRION": 2, "NSW": 100}
            },
            fw_name_constraint="structure optimization",
        )
    else:
        wf = add_modify_incar(
            wf,
            modify_incar_params={
                "incar_update": {"EDIFFG": 0.005, "IBRION": 2, "NSW": 100}
            },
            fw_name_constraint="structure optimization",
        )

    wf = add_modify_incar(
        wf,
        modify_incar_params={
            "incar_update": {"ADDGRID": ".TRUE.", "LASPH": ".TRUE.", "GGA": "PS"}
        },
    )

    # Includ ncl magmoms with saxis = (0, 0, 1)
    if magnetic and soc:
        magmoms = []
        for m in magmoms_orig:
            magmoms += [0.0, 0.0, m]
        magmoms = [str(m) for m in magmoms]
        magmoms = " ".join(magmoms)

        wf = add_modify_incar(
            wf,
            modify_incar_params={"incar_update": {"ISYM": 2, "MAGMOM": "%s" % magmoms}},
        )

    if magnetic and not soc:
        # Include magmoms in every calculation
        wf = add_modify_incar(
            wf,
            modify_incar_params={
                "incar_update": {"ISYM": 2, "MAGMOM": "%s" % magmoms, "ISPIN": 2}
            },
        )

    wf = add_modify_incar(
        wf,
        modify_incar_params={
            "incar_update": {
                "ISYM": 2,
                "LSORBIT": ".TRUE." if soc else ".FALSE.",
                "MAGMOM": "%s" % magmoms if magnetic else "%i*0.0" % ncoords,
                "ISPIN": 2 if magnetic and not soc else 1,
                "LWAVE": ".TRUE.",
                # "NBANDS": nbands,
            }
        },
        fw_name_constraint="nscf",
    )

    wf = add_common_powerups(wf, c)

    wf = add_additional_fields_to_taskdocs(
        wf, {"wf_meta": wf_meta}, task_name_constraint="VaspToDb"
    )

    if c.get("STABILITY_CHECK", STABILITY_CHECK):
        wf = add_stability_check(wf, fw_name_constraint="structure optimization")

    if c.get("ADD_WF_METADATA", ADD_WF_METADATA):
        wf = add_wf_metadata(wf, structure)

    return wf


def wf_vasp2trace_nonmagnetic(structure, c=None):
    """
    Fireworks workflow for running a vasp2trace calculation on a nonmagnetic material.

    Args:
        structure (Structure): Pymatgen structure object

    Returns:
        Workflow
    """

    c = c or {}
    vasp_cmd = c.get("VASP_CMD", VASP_CMD)
    db_file = c.get("DB_FILE", DB_FILE)

    ncoords = 3 * len(structure.sites)

    nbands = 0

    for site in structure.sites:
        nbands += site.species.total_electrons

    trim_kpoints = Kpoints(
        comment="TRIM Points",
        num_kpts=8,
        style=Kpoints.supported_modes.Reciprocal,
        kpts=(
            (0, 0, 0),
            (0.5, 0, 0),
            (0, 0.5, 0),
            (0, 0, 0.5),
            (0.5, 0.5, 0),
            (0, 0.5, 0.5),
            (0.5, 0, 0.5),
            (0.5, 0.5, 0.5),
        ),
        kpts_shift=(0, 0, 0),
        kpts_weights=[1, 1, 1, 1, 1, 1, 1, 1],
        coord_type="Reciprocal",
        labels=["gamma", "x", "y", "z", "s", "t", "u", "r"],
        tet_number=0,
        tet_weight=0,
        tet_connections=None,
    )

    wf = get_wf(
        structure,
        "vasp2trace_nonmagnetic.yaml",
        params=[
            {},
            {},
            {
                "input_set_overrides": {
                    "other_params": {"user_kpoints_settings": trim_kpoints}
                }
            },
            {},
        ],
        vis=MPStaticSet(structure, potcar_functional="PBE_54", force_gamma=True),
        common_params={"vasp_cmd": vasp_cmd, "db_file": db_file},
    )

    dim_data = StructureDimensionality(structure)

    if np.any(
        [
            dim == 2
            for dim in [dim_data.larsen_dim, dim_data.cheon_dim, dim_data.gorai_dim]
        ]
    ):
        wf = add_modify_incar(
            wf,
            modify_incar_params={
                "incar_update": {"IVDW": 11, "EDIFFG": 0.005, "IBRION": 2, "NSW": 100}
            },
            fw_name_constraint="structure optimization",
        )
    else:
        wf = add_modify_incar(
            wf,
            modify_incar_params={
                "incar_update": {"EDIFFG": 0.005, "IBRION": 2, "NSW": 100}
            },
            fw_name_constraint="structure optimization",
        )

    wf = add_modify_incar(
        wf,
        modify_incar_params={
            "incar_update": {"ADDGRID": ".TRUE.", "LASPH": ".TRUE.", "GGA": "PS"}
        },
    )

    wf = add_modify_incar(
        wf,
        modify_incar_params={
            "incar_update": {
                "ISYM": 2,
                "LSORBIT": ".TRUE.",
                "MAGMOM": "%i*0.0" % ncoords,
                "ISPIN": 1,
                "LWAVE": ".TRUE.",
                "NBANDS": nbands,
            }
        },
        fw_name_constraint="nscf",
    )

    wf = add_common_powerups(wf, c)

    if c.get("STABILITY_CHECK", STABILITY_CHECK):
        wf = add_stability_check(wf, fw_name_constraint="structure optimization")

    if c.get("ADD_WF_METADATA", ADD_WF_METADATA):
        wf = add_wf_metadata(wf, structure)

    return wf


def wf_vasp2trace_magnetic(structure, c=None):
    """
    Fireworks workflow for running a vasp2trace calculation 
    on a magnetic material.

    Args:
        structure (Structure): Pymatgen structure object with 
            magmom site property.

    Returns:
        Workflow

    """

    c = c or {}
    vasp_cmd = c.get("VASP_CMD", VASP_CMD)
    db_file = c.get("DB_FILE", DB_FILE)

    if "magmom" in structure.site_properties:
        magmoms = structure.site_properties["magmom"]
        magmoms = [str(m) for m in magmoms]
        magmoms = " ".join(magmoms)

    nbands = 0

    for site in structure.sites:
        nbands += site.species.total_electrons

    trim_kpoints = Kpoints(
        comment="TRIM Points",
        num_kpts=8,
        style=Kpoints.supported_modes.Reciprocal,
        kpts=(
            (0, 0, 0),
            (0.5, 0, 0),
            (0, 0.5, 0),
            (0, 0, 0.5),
            (0.5, 0.5, 0),
            (0, 0.5, 0.5),
            (0.5, 0, 0.5),
            (0.5, 0.5, 0.5),
        ),
        kpts_shift=(0, 0, 0),
        kpts_weights=[1, 1, 1, 1, 1, 1, 1, 1],
        coord_type="Reciprocal",
        labels=["gamma", "x", "y", "z", "s", "t", "u", "r"],
        tet_number=0,
        tet_weight=0,
        tet_connections=None,
    )

    wf_path = os.path.join(os.path.dirname(__file__), "vasp2trace_magnetic.yaml")
    wf = get_wf(
        structure,
        wf_path,
        params=[
            {},
            {},
            {
                "input_set_overrides": {
                    "other_params": {"user_kpoints_settings": trim_kpoints}
                }
            },
            {},
        ],
        vis=MPStaticSet(structure, potcar_functional="PBE_54", force_gamma=True),
        common_params={"vasp_cmd": vasp_cmd, "db_file": db_file},
    )

    dim_data = StructureDimensionality(structure)

    if np.any(
        [
            dim == 2
            for dim in [dim_data.larsen_dim, dim_data.cheon_dim, dim_data.gorai_dim]
        ]
    ):
        wf = add_modify_incar(
            wf,
            modify_incar_params={
                "incar_update": {"IVDW": 11, "EDIFFG": 0.005, "IBRION": 2, "NSW": 100}
            },
            fw_name_constraint="structure optimization",
        )
    else:
        wf = add_modify_incar(
            wf,
            modify_incar_params={
                "incar_update": {"EDIFFG": 0.005, "IBRION": 2, "NSW": 100}
            },
            fw_name_constraint="structure optimization",
        )

    wf = add_modify_incar(
        wf,
        modify_incar_params={
            "incar_update": {"ADDGRID": ".TRUE.", "LASPH": ".TRUE.", "GGA": "PS"}
        },
    )

    # Include magmoms in every calculation
    wf = add_modify_incar(
        wf,
        modify_incar_params={
            "incar_update": {"ISYM": 2, "MAGMOM": "%s" % magmoms, "ISPIN": 2}
        },
    )

    # Spin-polarized band structure
    wf = add_modify_incar(
        wf,
        modify_incar_params={
            "incar_update": {
                "ISYM": 2,
                "LSORBIT": ".FALSE.",
                "MAGMOM": "%s" % magmoms,
                "ISPIN": 2,
                "LWAVE": ".TRUE.",
                "NBANDS": nbands,
            }
        },
        fw_name_constraint="nscf",
    )

    wf = add_common_powerups(wf, c)

    if c.get("STABILITY_CHECK", STABILITY_CHECK):
        wf = add_stability_check(wf, fw_name_constraint="structure optimization")

    if c.get("ADD_WF_METADATA", ADD_WF_METADATA):
        wf = add_wf_metadata(wf, structure)

    return wf


class Z2PackWF:
    def __init__(
        self, structure, symmetry_reduction=True, vasp_cmd=VASP_CMD, db_file=DB_FILE
    ):
        """
        Fireworks workflow for running Z2Pack to compute Z2 invariants and Chern numbers.

        ***VASP_CMD in my_fworker.yaml MUST be set to "vasp_ncl" for Z2Pack.

        Args:
            structure (Structure): Pymatgen structure object
            symmetry_reduction (bool): Set to False to disable symmetry reduction and 
            include all 6 BZ surfaces (for magnetic systems).

        """

        self.structure = structure
        self.symmetry_reduction = symmetry_reduction
        self.uuid = str(uuid4())
        self.wf_meta = {"wf_uuid": self.uuid, "wf_name": "Z2Pack WF"}

    @staticmethod
    def _get_reciprocal_point_group_nonmagnetic(struct):
        """
        Obtain the symmetry ops. in the reciprocal point group of an input structure.  

        Returns:
            recip_point_group (list): List of symmetry operations as numpy arrays in the 
            fractional reciprocal space basis. 

        """
        R = -1 * np.eye(3)
        sga = SpacegroupAnalyzer(struct)
        ops = sga.get_symmetry_operations()
        isomorphic_point_group = [op.rotation_matrix for op in ops]

        V = struct.lattice.matrix.T  # fractional real space to cartesian real space
        # fractional reciprocal space to cartesian reciprocal space
        W = struct.lattice.reciprocal_lattice.matrix.T
        # fractional real space to fractional reciprocal space
        A = np.dot(np.linalg.inv(W), V)

        Ainv = np.linalg.inv(A)
        # convert to reciprocal primitive basis
        recip_point_group = [np.around(np.dot(A, np.dot(R, Ainv)), decimals=2)]
        for op in isomorphic_point_group:
            op = np.around(np.dot(A, np.dot(op, Ainv)), decimals=2)
            new = True
            new_coset = True
            for thing in recip_point_group:
                if (thing == op).all():
                    new = False
                if (thing == np.dot(R, op)).all():
                    new_coset = False

            if new:
                recip_point_group.append(op)
            if new_coset:
                recip_point_group.append(np.dot(R, op))

        return recip_point_group

    @staticmethod
    def _is_permutation_eq(A, B):
        """
        Check for equivalency between two arrays including permutations. 

        Returns:
            Whether the two arrays are equivalent (True) or not (False). 

        """
        count = {}
        for a in A:
            count[str(a)] = 1

        for b in B:
            if str(b) in count:
                if count[str(b)] == 0:
                    return False
                else:
                    count[str(b)] = count[str(b)] - 1
            else:
                return False

        return True

    def get_equiv_planes(self):
        """
        Get equivalent TRIM planes in the BZ using the reciprocal point symmetry.

        Returns:
            plane_equiv (dict): Dictionary providing equivalent TRIM plane names. 

        """
        struct = self.structure
        rpg_ops = Z2PackWF._get_reciprocal_point_group_nonmagnetic(struct)

        trim_pts = list(itertools.product((0.0, 0.5), repeat=3))

        planes = {}
        plane_equiv = {}
        symbols = ["x", "y", "z"]
        for coord in range(3):
            for plane_num in range(2):
                planes["k%s_%s" % (symbols[coord], str(plane_num))] = [
                    np.array(pt)
                    for pt in trim_pts
                    if pt[coord] == float(plane_num) / 2.0
                ]
                plane_equiv["k%s_%s" % (symbols[coord], str(plane_num))] = []

        for plane in planes.keys():
            for op in rpg_ops:
                trans_pts = [np.dot(op, pt) % 1.0 for pt in planes[plane]]

                for other_plane in planes.keys():
                    if other_plane != plane:
                        check_eq = Z2PackWF._is_permutation_eq(
                            planes[other_plane], trans_pts
                        )

                        if check_eq and other_plane not in plane_equiv[plane]:
                            plane_equiv[plane].append(other_plane)

        return plane_equiv

    def get_wf(self, c=None):
        """
        Get the workflow.

        Returns:
            Workflow

        """

        c = c or {"VASP_CMD": VASP_CMD, "DB_FILE": DB_FILE}

        vis = MPRelaxSet(self.structure, potcar_functional="PBE_54", force_gamma=True)

        opt_fw = OptimizeFW(
            self.structure,
            vasp_input_set=vis,
            vasp_cmd=c["VASP_CMD"],
            db_file=c["DB_FILE"],
        )

        vis = MPStaticSet(self.structure, potcar_functional="PBE_54", force_gamma=True)

        static_fw = StaticFW(
            self.structure,
            vasp_input_set=vis,
            vasp_cmd=c["VASP_CMD"],
            db_file=c["DB_FILE"],
            parents=[opt_fw],
        )

        # Separate FW for each BZ surface calc
        # Run Z2Pack on unique TRIM planes in the BZ

        surfaces = ["kx_0", "kx_1"]
        equiv_planes = self.get_equiv_planes()

        # Only run calcs on inequivalent BZ surfaces
        if self.symmetry_reduction:
            for add_surface in equiv_planes.keys():
                mark = True
                for surface in surfaces:
                    if surface in equiv_planes[add_surface]:
                        mark = False
                if mark and add_surface not in surfaces:
                    surfaces.append(add_surface)
        else:  # 4 TRI surfaces define Z2 in 3D
            surfaces = ["kx_1", "ky_1", "kz_0", "kz_1"]

        z2pack_fws = []

        for surface in surfaces:
            z2pack_fw = Z2PackFW(
                parents=[static_fw],
                structure=self.structure,
                surface=surface,
                uuid=self.uuid,
                name="z2pack",
                vasp_cmd=c["VASP_CMD"],
                db_file=c["DB_FILE"],
            )
            z2pack_fws.append(z2pack_fw)

        analysis_fw = InvariantFW(
            parents=z2pack_fws,
            structure=self.structure,
            symmetry_reduction=self.symmetry_reduction,
            equiv_planes=equiv_planes,
            uuid=self.uuid,
            name="invariant",
            db_file=c["DB_FILE"],
        )

        fws = [opt_fw, static_fw] + z2pack_fws + [analysis_fw]

        wf = Workflow(fws)
        wf = add_additional_fields_to_taskdocs(wf, {"wf_meta": self.wf_meta})

        # Add vdW corrections if structure is layered
        dim_data = StructureDimensionality(self.structure)

        if np.any(
            [
                dim == 2
                for dim in [dim_data.larsen_dim, dim_data.cheon_dim, dim_data.gorai_dim]
            ]
        ):
            wf = add_modify_incar(
                wf,
                modify_incar_params={
                    "incar_update": {
                        "IVDW": 11,
                        "EDIFFG": 0.005,
                        "IBRION": 2,
                        "NSW": 100,
                    }
                },
                fw_name_constraint="structure optimization",
            )

            wf = add_modify_incar(
                wf,
                modify_incar_params={"incar_update": {"IVDW": 11}},
                fw_name_constraint="static",
            )

            wf = add_modify_incar(
                wf,
                modify_incar_params={"incar_update": {"IVDW": 11}},
                fw_name_constraint="z2pack",
            )

        else:
            wf = add_modify_incar(
                wf,
                modify_incar_params={
                    "incar_update": {"EDIFFG": 0.005, "IBRION": 2, "NSW": 100}
                },
                fw_name_constraint="structure optimization",
            )

        # Helpful vasp settings and no parallelization
        wf = add_modify_incar(
            wf,
            modify_incar_params={
                "incar_update": {
                    "ADDGRID": ".TRUE.",
                    "LASPH": ".TRUE.",
                    "GGA": "PS",
                    "NCORE": 1,
                }
            },
        )

        # Generate inputs for Z2Pack with a static calc
        wf = add_modify_incar(
            wf,
            modify_incar_params={"incar_update": {"PREC": "Accurate"}},
            fw_name_constraint="static",
        )

        wf = add_common_powerups(wf, c)

        wf.name = "{} {}".format(self.structure.composition.reduced_formula, "Z2Pack")

        if c.get("STABILITY_CHECK", STABILITY_CHECK):
            wf = add_stability_check(wf, fw_name_constraint="structure optimization")

        if c.get("ADD_WF_METADATA", ADD_WF_METADATA):
            wf = add_wf_metadata(wf, self.structure)

        tag = "z2pack: {}".format(self.uuid)
        wf = add_tags(wf, [tag])

        return wf
