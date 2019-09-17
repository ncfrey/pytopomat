import warnings

from fireworks import Firework

from pymatgen import Structure

from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.common.firetasks.glue_tasks import PassCalcLocs, CopyFiles
from atomate.vasp.firetasks.parse_outputs import VaspToDb

from pytopomat.workflows.firetasks import (
    Vasp2TraceToDb,
    RunVasp2Trace,
    CopyVaspOutputs,
    Z2PackToDb,
    RunZ2Pack,
    WriteWannier90Win,
)


class Vasp2TraceFW(Firework):
    def __init__(
        self,
        parents=None,
        structure=None,
        name="vasp2trace",
        db_file=None,
        prev_calc_dir=None,
        vasp2trace_out=None,
        vasp_cmd=None,
        **kwargs
    ):
        """
        Run Vasp2Trace and parse the output data. Assumes you have a previous FW with the 
        calc_locs passed into the current FW.

        Args:
            structure (Structure): - only used for setting name of FW
            name (str): name of this FW
            db_file (str): path to the db file
            parents (Firework): Parents of this particular Firework. FW or list of FWS.
            prev_calc_dir (str): Path to a previous calculation to copy from
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        fw_name = "{}-{}".format(
            structure.composition.reduced_formula if structure else "unknown", name
        )

        t = []

        if prev_calc_dir:
            t.append(
                CopyVaspOutputs(
                    calc_dir=prev_calc_dir,
                    additional_files=["CHGCAR", "WAVECAR"],
                    contcar_to_poscar=True,
                )
            )
        elif parents:
            t.append(
                CopyVaspOutputs(
                    calc_loc=True,
                    additional_files=["CHGCAR", "WAVECAR"],
                    contcar_to_poscar=True,
                )
            )
        else:
            raise ValueError("Must specify structure or previous calculation")

        t.extend(
            [
                RunVasp2Trace(),
                PassCalcLocs(name=name),
                Vasp2TraceToDb(db_file=db_file, vasp2trace_out=vasp2trace_out),
            ]
        )

        super(Vasp2TraceFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


class Z2PackFW(Firework):
    def __init__(
        self,
        parents=None,
        structure=None,
        uuid=None,
        name="z2pack",
        db_file=None,
        prev_calc_dir=None,
        z2pack_out=None,
        vasp_cmd=None,
        **kwargs
    ):
        """
        Run Z2Pack and parse the output data. Assumes you have a previous FW with the calc_locs passed into the current FW.

        Args:
            structure (Structure): Structure object.
            uuid (str): Unique wf identifier.
            name (str): name of this FW
            db_file (str): path to the db file
            parents (Firework): Parents of this particular Firework. FW or list of FWS.
            prev_calc_dir (str): Path to a previous calculation to copy from
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        fw_name = "{}-{}".format(
            structure.composition.reduced_formula if structure else "unknown", name
        )

        t = []

        if prev_calc_dir:
            t.append(
                CopyVaspOutputs(
                    calc_dir=prev_calc_dir,
                    additional_files=["CHGCAR"],
                    contcar_to_poscar=True,
                )
            )
        elif parents:
            t.append(
                CopyVaspOutputs(
                    calc_loc=True, additional_files=["CHGCAR"], contcar_to_poscar=True
                )
            )
        else:
            raise ValueError("Must specify structure or previous calculation")

        t.append(WriteWannier90Win(wf_uuid=uuid, db_file=db_file))

        # Copy files to a folder called 'input' for z2pack
        files_to_copy = ['CHGCAR', 'INCAR', 'POSCAR', 'POTCAR', 'wannier90.win']
        t.append(CopyFiles(from_dir='./', to_dir='input', files_to_copy=files_to_copy))

        # Run Z2Pack on 6 TRI planes in the BZ
        surfaces = [
            lambda s, t: [0, s / 2, t],
            lambda s, t: [0.5, s / 2, t],
            lambda s, t: [s / 2, 0, t],
            lambda s, t: [s / 2, 0.5, t],
            lambda s, t: [s / 2, t, 0],
            lambda s, t: [s / 2, t, 0.5],
        ]

        surface_labels = ['kx_0', 'kx_1', 'ky_0', 'ky_1', 'kz_0', 'kz_1']

        for surface, surface_label in zip(surfaces, surface_labels):
            t.append(RunZ2Pack(surface=surface, surface_label=surface_label))

        t.extend(
            [
                PassCalcLocs(name=name),
                Z2PackToDb(db_file=db_file, z2pack_out=z2pack_out),
            ]
        )

        super().__init__(t, parents=parents, name=fw_name, **kwargs)
