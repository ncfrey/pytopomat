import warnings
import os

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
    SetUpZ2Pack,
    RunZ2Pack,
    WriteWannier90Win,
    InvariantsToDB,
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
                    additional_files=["CHGCAR", "WAVECAR", "PROCAR"],
                    contcar_to_poscar=True,
                )
            )
        elif parents:
            t.append(
                CopyVaspOutputs(
                    calc_loc=True,
                    additional_files=["CHGCAR", "WAVECAR", "PROCAR"],
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


class Vasp2TraceMagneticFW(Firework):
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
        Run Vasp2Trace on a spin-polarized calculation and parse the output data. 
        Assumes you have a previous FW with the calc_locs passed into the current FW.

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
                    additional_files=["CHGCAR", "WAVECAR", "PROCAR"],
                    contcar_to_poscar=True,
                )
            )
        elif parents:
            t.append(
                CopyVaspOutputs(
                    calc_loc=True,
                    additional_files=["CHGCAR", "WAVECAR", "PROCAR"],
                    contcar_to_poscar=True,
                )
            )
        else:
            raise ValueError("Must specify structure or previous calculation")

        t.extend(
            [
                RunVasp2TraceMagnetic(),
                PassCalcLocs(name=name),
                Vasp2TraceToDb(db_file=db_file, vasp2trace_out=vasp2trace_out),
            ]
        )

        super(Vasp2TraceMagneticFW, self).__init__(
            t, parents=parents, name=fw_name, **kwargs
        )


class Z2PackFW(Firework):
    def __init__(
        self,
        parents=None,
        structure=None,
        surface=None,
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
            surface (str): Like "kx_0", "kx_1", "ky_0", etc. that indicates TRIM surface in BZ.
            uuid (str): Unique wf identifier.
            name (str): name of this FW
            db_file (str): path to the db file
            parents (Firework): Parents of this particular Firework. FW or list of FWS.
            prev_calc_dir (str): Path to a previous calculation to copy from
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """

        self.structure = structure
        nsites = len(self.structure.sites)

        # Check for magmoms
        if "magmom" in self.structure.site_properties:
            l = [[0.0, 0.0, m] for m in self.structure.site_properties["magmom"]]
            ncl_magmoms = [elem for ll in l for elem in ll]
        else:
            ncl_magmoms = 3 * nsites * [0.0]

        ncl_magmoms = [str(m) for m in ncl_magmoms]
        ncl_magmoms = " ".join(ncl_magmoms)

        fw_name = "{}-{}".format(
            structure.composition.reduced_formula if structure else "unknown", surface
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
        t.append(SetUpZ2Pack(ncl_magmoms=ncl_magmoms, wf_uuid=uuid, db_file=db_file))

        t.append(RunZ2Pack(surface=surface))

        t.extend([PassCalcLocs(name=name), Z2PackToDb(db_file=db_file, wf_uuid=uuid)])

        super().__init__(t, parents=parents, name=fw_name, **kwargs)


class InvariantFW(Firework):
    def __init__(
        self,
        parents=None,
        structure=None,
        symmetry_reduction=None,
        equiv_planes=None,
        uuid=None,
        name="invariant",
        db_file=None,
        **kwargs
    ):
        """
        Process Z2Pack outputs, e.g. calculate Z2=(v0; v1, v2, v3).

        Args:
            parents (list): Parent FWs.
            structure (Structure): Structure object.
            symmetry_reduction (bool): Set to False to disable symmetry reduction and 
            include all 6 BZ surfaces (for magnetic systems).
            equiv_planes (list): Like "kx_0", "kx_1", "ky_0", etc. that indicates TRIM surface in BZ.
            uuid (str): Unique wf identifier.
            name (str): name of this FW
            db_file (str): path to the db file
            parents (Firework): Parents of this particular Firework. FW or list of FWS.
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """

        fw_name = "{}-{}".format(
            structure.composition.reduced_formula if structure else "unknown",
            "invariant",
        )

        t = []

        # Create a dictionary of TRIM surface: Z2 invariant, Chern number
        t.append(
            InvariantsToDB(
                wf_uuid=uuid,
                db_file=db_file,
                structure=structure,
                symmetry_reduction=symmetry_reduction,
                equiv_planes=equiv_planes,
            )
        )

        super().__init__(t, parents=parents, name=fw_name, **kwargs)
