import numpy as np

from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.sets import MPStaticSet

from atomate.vasp.config import STABILITY_CHECK, VASP_CMD, DB_FILE, ADD_WF_METADATA
from atomate.vasp.powerups import add_stability_check, add_modify_incar, \
    add_wf_metadata, add_common_powerups
from atomate.vasp.workflows.base.core import get_wf

from pytopomat.analyzer import StructureDimensionality


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

    ncoords = 3*len(structure.sites)

    nbands = 0

    for site in structure.sites:
        nbands += site.species.total_electrons

    trim_kpoints = Kpoints(comment="TRIM Points", num_kpts=8,
                           style=Kpoints.supported_modes.Reciprocal,
                           kpts=((0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5),
                                 (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0.5)),
                           kpts_shift=(0, 0, 0),
                           kpts_weights=[1, 1, 1, 1, 1, 1, 1, 1],
                           coord_type='Reciprocal',
                           labels=['gamma', 'x', 'y', 'z', 's', 't', 'u', 'r'],
                           tet_number=0, tet_weight=0, tet_connections=None)

    wf = get_wf(structure, "vasp2trace_nonmagnetic.yaml",
                params=[{}, {}, {'input_set_overrides': {
                    'other_params': {'user_kpoints_settings': trim_kpoints}}}, {}],
                vis=MPStaticSet(structure, force_gamma=True),
                common_params={"vasp_cmd": vasp_cmd, "db_file": db_file})

    dim_data = StructureDimensionality(structure)

    if np.any([dim == 2 for dim in [dim_data.larsen_dim, dim_data.cheon_dim, dim_data.gorai_dim]]):
        wf = add_modify_incar(wf,
                              modify_incar_params={'incar_update': {
                                  'IVDW': 11, 'EDIFFG': 0.005, 'IBRION': 2, 'NSW': 100}},
                              fw_name_constraint='structure optimization')
    else:
        wf = add_modify_incar(wf,
                              modify_incar_params={
                                  'incar_update': {'EDIFFG': 0.005, 'IBRION': 2, 'NSW': 100}},
                              fw_name_constraint='structure optimization')

    wf = add_modify_incar(wf,
                          modify_incar_params={'incar_update': {'ADDGRID': '.TRUE.', 'LASPH': '.TRUE.', 'GGA': 'PS'}})

    wf = add_modify_incar(wf,
                          modify_incar_params={'incar_update': {'ISYM': 2, 'LSORBIT': '.TRUE.',
                                                                'MAGMOM': '%i*0.0' % ncoords, 'ISPIN': 1, 'LWAVE': '.TRUE.',
                                                                'NBANDS': nbands}}, fw_name_constraint='nscf')

    wf = add_common_powerups(wf, c)

    if c.get("STABILITY_CHECK", STABILITY_CHECK):
        wf = add_stability_check(
            wf, fw_name_constraint="structure optimization")

    if c.get("ADD_WF_METADATA", ADD_WF_METADATA):
        wf = add_wf_metadata(wf, structure)

    return wf
