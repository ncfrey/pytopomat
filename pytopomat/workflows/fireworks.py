import warnings

from fireworks import Firework

from pymatgen import Structure

from atomate.vasp.config import VASP_CMD, DB_FILE
from atomate.common.firetasks.glue_tasks import PassCalcLocs
from atomate.vasp.firetasks.parse_outputs import VaspToDb

from pytopomat.workflows.firetasks import Vasp2TraceToDb, RunVasp2Trace, CopyVaspOutputs


class Vasp2TraceFW(Firework):

    def __init__(self, parents=None, structure=None, name="vasp2trace", db_file=None,
                 prev_calc_dir=None, vasp2trace_out=None, vasp_cmd=None, **kwargs):
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
            structure.composition.reduced_formula if structure else "unknown", name)

        t = []

        if prev_calc_dir:
            t.append(CopyVaspOutputs(calc_dir=prev_calc_dir, additional_files=["CHGCAR", "WAVECAR"],
                                     contcar_to_poscar=True))
        elif parents:
            t.append(CopyVaspOutputs(calc_loc=True, additional_files=["CHGCAR", "WAVECAR"],
                                     contcar_to_poscar=True))
        else:
            raise ValueError("Must specify structure or previous calculation")

        t.extend([RunVasp2Trace(),
                  PassCalcLocs(name=name),
                  Vasp2TraceToDb(db_file=db_file,
                                 vasp2trace_out=vasp2trace_out)])

        super(Vasp2TraceFW, self).__init__(
            t, parents=parents, name=fw_name, **kwargs)
