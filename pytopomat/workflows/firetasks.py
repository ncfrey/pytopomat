"""
Firetasks for FWs.

"""

import shutil
import json
import os
import numpy as np

from monty.json import MontyEncoder, jsanitize

from spglib import standardize_cell

from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Incar, Outcar

from pytopomat.irvsp_caller import IRVSPCaller, IRVSPOutput
from pytopomat.vasp2trace_caller import (
    Vasp2TraceCaller,
    Vasp2Trace2Caller,
    Vasp2TraceOutput,
)
from pytopomat.z2pack_caller import Z2PackCaller

from fireworks import explicit_serialize, FiretaskBase, FWAction
from fireworks.utilities.fw_serializers import DATETIME_HANDLER

from atomate.utils.utils import env_chk, get_logger
from atomate.vasp.database import VaspCalcDb

logger = get_logger(__name__)


@explicit_serialize
class RunIRVSP(FiretaskBase):
    """
    Execute IRVSP in current directory.

    """

    def run_task(self, fw_spec):

        wd = os.getcwd()
        IRVSPCaller(wd)

        try:
            raw_struct = Structure.from_file(wd + "/POSCAR")
            formula = raw_struct.composition.formula
            structure = raw_struct.as_dict()

            outcar = Outcar(wd + "/OUTCAR")
            efermi = outcar.efermi

        except:
            formula = None
            structure = None
            efermi = None

        data = IRVSPOutput(wd + "/outir.txt")

        return FWAction(
            update_spec={
                "irvsp_out": data.as_dict(),
                "structure": structure,
                "formula": formula,
                "efermi": efermi,
            }
        )


@explicit_serialize
class StandardizeCell(FiretaskBase):
    """
    Standardize cell with spglib and symprec=1e-2.

    """

    def run_task(self, fw_spec):

        wd = os.getcwd()

        struct = Structure.from_file(wd + "/POSCAR")

        numbers = [site.specie.number for site in struct]
        lattice = struct.lattice.matrix
        positions = struct.frac_coords

        if "magmom" in struct.site_properties:
            magmoms = struct.site_properties["magmom"]
            cell = (lattice, positions, numbers, magmoms)
        else:
            magmoms = None
            cell = (lattice, positions, numbers)

        lat, pos, nums = standardize_cell(cell, to_primitive=False, symprec=1e-2)

        structure = Structure(lat, nums, pos)

        if magmoms is not None:
            structure.add_site_property("magmom", magmoms)

        structure.to(fmt="poscar", filename="CONTCAR")

        return FWAction(update_spec={"structure": structure})


@explicit_serialize
class IRVSPToDb(FiretaskBase):
    """
    Stores data from outir.txt that is output by irvsp.

    required_params:
        irvsp_out (IRVSPOutput): output from IRVSP calculation.

    optional_params:
        db_file (str): path to the db file
        additional_fields (dict): dict of additional fields to add
        
    """

    required_params = ["irvsp_out"]
    optional_params = ["db_file", "additional_fields"]

    def run_task(self, fw_spec):

        irvsp = self["irvsp_out"] or fw_spec["irvsp_out"]

        irvsp = jsanitize(irvsp)

        additional_fields = self.get("additional_fields", {})
        d = additional_fields.copy()
        d["formula"] = fw_spec["formula"]
        d["efermi"] = fw_spec["efermi"]
        d["structure"] = fw_spec["structure"]
        d["irvsp"] = irvsp

        # store the results
        db_file = env_chk(self.get("db_file"), fw_spec)
        if not db_file:
            with open("irvsp.json", "w") as f:
                f.write(json.dumps(d, default=DATETIME_HANDLER))
        else:
            db = VaspCalcDb.from_db_file(db_file, admin=True)
            db.collection = db.db["irvsp"]
            db.collection.insert_one(d)
            logger.info("IRVSP calculation complete.")
        return FWAction()


@explicit_serialize
class Vasp2TraceToDb(FiretaskBase):
    """
    Stores data from traces.txt that is output by vasp2trace.

    optional_params:
        db_file (str): path to the db file
    """

    required_params = ["vasp2trace_out"]
    optional_params = ["db_file"]

    def run_task(self, fw_spec):

        v2t = self["vasp2trace_out"] or fw_spec["vasp2trace_out"]

        v2t = jsanitize(v2t)

        d = {
            "formula": fw_spec["formula"],
            "structure": fw_spec["structure"],
            "vasp2trace": v2t,
        }

        # store the results
        db_file = env_chk(self.get("db_file"), fw_spec)
        if not db_file:
            with open("vasp2trace.json", "w") as f:
                f.write(json.dumps(d, default=DATETIME_HANDLER))
        else:
            db = VaspCalcDb.from_db_file(db_file, admin=True)
            db.collection = db.db["vasp2trace"]
            db.collection.insert_one(d)
            logger.info("Vasp2trace calculation complete.")
        return FWAction()


@explicit_serialize
class RunVasp2Trace(FiretaskBase):
    """
    Execute vasp2trace in current directory.

    """

    def run_task(self, fw_spec):

        wd = os.getcwd()
        Vasp2TraceCaller(wd)

        try:
            raw_struct = Structure.from_file(wd + "/POSCAR")
            formula = raw_struct.composition.formula
            structure = raw_struct.as_dict()

        except:
            formula = None
            structure = None

        data = Vasp2TraceOutput(wd + "/trace.txt")

        return FWAction(
            update_spec={
                "vasp2trace_out": data.as_dict(),
                "structure": structure,
                "formula": formula,
            }
        )


@explicit_serialize
class RunVasp2TraceMagnetic(FiretaskBase):
    """
    Execute vasp2trace in current directory with spin-polarized calculation.

    """

    def run_task(self, fw_spec):

        wd = os.getcwd()
        Vasp2Trace2Caller(wd)  # version2 of vasp2trace for spin-polarized calcs

        try:
            raw_struct = Structure.from_file(wd + "/POSCAR")
            formula = raw_struct.composition.formula
            structure = raw_struct.as_dict()

        except:
            composition = None
            structure = None

        up_data = Vasp2TraceOutput(wd + "/trace_up.txt")
        down_data = Vasp2TraceOutput(wd + "/trace_dn.txt")

        return FWAction(
            update_spec={
                "vasp2trace_out": {
                    "up": up_data.as_dict(),
                    "down": down_data.as_dict(),
                },
                "structure": structure,
                "formula": formula,
            }
        )


@explicit_serialize
class SetUpZ2Pack(FiretaskBase):
    """
    Set up input files for a z2pack run.

    required_params:
        ncl_magmoms (str): 3*natoms long array of x,y,z magmoms for each ion.

    """

    required_params = ["ncl_magmoms", "wf_uuid", "db_file"]

    def run_task(self, fw_spec):

        ncl_magmoms = self["ncl_magmoms"]

        # Get num of electrons and bands from static calc
        uuid = self["wf_uuid"]
        db_file = env_chk(self.get("db_file"), fw_spec)
        db = VaspCalcDb.from_db_file(db_file, admin=True)
        db.collection = db.db["tasks"]

        task_doc = db.collection.find_one(
            {"wf_meta.wf_uuid": uuid, "task_label": "static"}, ["input.parameters"]
        )

        nelec = int(task_doc["input"]["parameters"]["NELECT"])
        nbands = int(task_doc["input"]["parameters"]["NBANDS"])

        incar = Incar.from_file("INCAR")

        # Modify INCAR for Z2Pack
        incar_update = {
            "PREC": "Accurate",
            "LSORBIT": ".TRUE.",
            "GGA_COMPAT": ".FALSE.",
            "LASPH": ".TRUE.",
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "ISYM": -1,
            "LPEAD": ".FALSE.",
            "LWANNIER90": ".TRUE.",
            "LWRITE_MMN_AMN": ".TRUE.",
            "LWAVE": ".FALSE.",
            "ICHARG": 11,
            "MAGMOM": "%s" % ncl_magmoms,
            "NBANDS": "%d" % (2 * nbands),
        }

        incar.update(incar_update)
        incar.write_file("INCAR")

        try:
            struct = Structure.from_file("POSCAR")
            formula = struct.composition.formula
            reduced_formula = struct.composition.reduced_formula
            structure = struct.as_dict()

        except:
            formula = None
            structure = None
            reduced_formula = None

        files_to_copy = ["CHGCAR", "INCAR", "POSCAR", "POTCAR", "wannier90.win"]

        os.mkdir("input")
        for file in files_to_copy:
            shutil.move(file, "input")

        return FWAction(
            update_spec={
                "structure": structure,
                "formula": formula,
                "reduced_formula": reduced_formula,
            }
        )


@explicit_serialize
class RunZ2Pack(FiretaskBase):
    """
    Call Z2Pack.

    required_params:
        surface (str): TRIM surface, e.g. k_x = 0 or k_x = 1/2.

    """

    required_params = ["surface"]

    def run_task(self, fw_spec):

        z2pc = Z2PackCaller(input_dir="input", surface=self["surface"])

        z2pc.run(z2_settings=None)

        data = z2pc.output

        return FWAction(update_spec={self["surface"]: data.as_dict()})


@explicit_serialize
class Z2PackToDb(FiretaskBase):
    """
    Stores data from running Z2Pack.

    optional_params:
        db_file (str): path to the db file
    """

    optional_params = ["db_file", "wf_uuid"]

    def run_task(self, fw_spec):

        wf_uuid = self["wf_uuid"]

        surfaces = ["kx_0", "kx_1", "ky_0", "ky_1", "kz_0", "kz_1"]

        d = {
            "wf_uuid": wf_uuid,
            "formula": fw_spec["formula"],
            "reduced_formula": fw_spec["reduced_formula"],
            "structure": fw_spec["structure"],
        }

        for surface in surfaces:
            if surface in fw_spec.keys():
                d[surface] = fw_spec[surface]

        d = jsanitize(d)

        # store the results
        db_file = env_chk(self.get("db_file"), fw_spec)
        if not db_file:
            with open("z2pack.json", "w") as f:
                f.write(json.dumps(d, default=DATETIME_HANDLER))
        else:
            db = VaspCalcDb.from_db_file(db_file, admin=True)
            db.collection = db.db["z2pack"]
            db.collection.insert_one(d)
            logger.info("Z2Pack surface calculation complete.")

        return FWAction()


@explicit_serialize
class WriteWannier90Win(FiretaskBase):
    """
    Write the wannier90.win input file for Z2Pack.

    required_params:
        wf_uuid (str): Unique identifier
        db_file (str): path to the db file
    """

    required_params = ["wf_uuid", "db_file"]

    def run_task(self, fw_spec):

        # Get num of electrons and bands from static calc
        uuid = self["wf_uuid"]
        db_file = env_chk(self.get("db_file"), fw_spec)
        db = VaspCalcDb.from_db_file(db_file, admin=True)
        db.collection = db.db["tasks"]

        task_doc = db.collection.find_one(
            {"wf_meta.wf_uuid": uuid, "task_label": "static"}, ["input.parameters"]
        )

        nelec = int(task_doc["input"]["parameters"]["NELECT"])
        nbands = int(task_doc["input"]["parameters"]["NBANDS"])

        w90_file = [
            "num_wann = %d" % (nelec),
            "num_bands = %d" % (nelec),  # 1 band / elec with SOC
            "spinors=.true.",
            "num_iter 0",
            "shell_list 1",
            "exclude_bands %d-%d" % (nelec + 1, 2 * nbands),
        ]

        w90_file = "\n".join(w90_file)

        with open("wannier90.win", "w") as f:
            f.write(w90_file)

        return FWAction()


@explicit_serialize
class InvariantsToDB(FiretaskBase):
    """
    Store Z2 and Chern nums on TRIM surfaces from Z2P output.

    required_params:
        wf_uuid (str): Unique wf identifier.
        symmetry_reduction (bool): Set to False to disable symmetry reduction
            and include all 6 BZ surfaces (for magnetic systems).
        equiv_planes (dict): of the form {kx_0': ['ky_0', 'kz_0']}.

    """

    required_params = [
        "wf_uuid",
        "db_file",
        "structure",
        "symmetry_reduction",
        "equiv_planes",
    ]

    def run_task(self, fw_spec):

        surfaces = ["kx_0", "kx_1", "ky_0", "ky_1", "kz_0", "kz_1"]
        structure = self["structure"]
        symmetry_reduction = self["symmetry_reduction"]
        equiv_planes = self["equiv_planes"]

        # Get invariants for each surface
        uuid = self["wf_uuid"]
        db_file = env_chk(self.get("db_file"), fw_spec)
        db = VaspCalcDb.from_db_file(db_file, admin=True)
        db.collection = db.db["z2pack"]

        task_docs = db.collection.find({"wf_uuid": uuid})

        z2_dict = {}
        chern_dict = {}
        for doc in task_docs:
            for s in surfaces:
                if s in doc.keys():
                    z2_dict[s] = doc[s]["z2_invariant"]
                    chern_dict[s] = doc[s]["chern_number"]

        # Write invariants for equivalent planes
        if symmetry_reduction and len(z2_dict) < 6:  # some equivalent planes
            for surface in equiv_planes.keys():
                # Z2
                if surface in z2_dict.keys() and len(equiv_planes[surface]) > 0:
                    for ep in equiv_planes[surface]:
                        if ep not in z2_dict.keys():
                            z2_dict[ep] = z2_dict[surface]
                # Chern
                if surface in chern_dict.keys() and len(equiv_planes[surface]) > 0:
                    for ep in equiv_planes[surface]:
                        if ep not in chern_dict.keys():
                            chern_dict[ep] = chern_dict[surface]

        # Compute Z2 invariant
        if all(
            surface in z2_dict.keys() for surface in ["kx_1", "ky_1", "kz_0", "kz_1"]
        ):
            v0 = (z2_dict["kz_0"] + z2_dict["kz_1"]) % 2
            v1 = z2_dict["kx_1"]
            v2 = z2_dict["ky_1"]
            v3 = z2_dict["kz_1"]
            z2 = (v0, v1, v2, v3)
        else:
            z2 = (np.nan, np.nan, np.nan, np.nan)

        # store the results
        d = {
            "wf_uuid": uuid,
            "task_label": "topological invariants",
            "formula": structure.composition.formula,
            "reduced_formula": structure.composition.reduced_formula,
            "structure": structure.as_dict(),
            "z2_dict": z2_dict,
            "chern_dict": chern_dict,
            "z2": z2,
            "equiv_planes": equiv_planes,
            "symmetry_reduction": symmetry_reduction,
        }

        d = jsanitize(d)

        db.collection.insert_one(d)

        return FWAction()
