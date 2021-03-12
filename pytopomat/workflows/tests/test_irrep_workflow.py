import os
import pytest

from pymatgen.core import Structure, Lattice

from fireworks import LaunchPad
from atomate.utils.testing import DB_DIR
from atomate.vasp.database import VaspCalcDb

from pytopomat.workflows.core import wf_irrep
from pytopomat.workflows.firetasks import IrrepToDb

from pytopomat.irrep_caller import IrrepOutput

module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
db_dir = os.path.join(module_dir, "..", "common", "test_files")
test_dir = os.path.join(module_dir, "..", "..", "..", "test_files")


class TestIrrepWorkflow(object):
    @pytest.fixture
    def no_connection(self):
        """Check for connection to local MongoDB."""

        try:
            lp = LaunchPad.from_file(os.path.join(DB_DIR, "my_launchpad.yaml"))
            lp.reset("", require_password=False)
            return False
        except:
            return True

    @pytest.fixture
    def bi(self):
        """Return BCC Bi structure."""

        bcc_bi = Structure.from_spacegroup(
            "Im-3m", Lattice.cubic(3.453), ["Bi"], [[0, 0, 0]]
        )
        bcc_bi = bcc_bi.get_reduced_structure("niggli")
        return bcc_bi

    # Will fail without .yml specs in atomate library
    def test_get_wflow(self, bi):

        wf = wf_irrep(bi, magnetic=False, soc=True)
        wf_dict = wf.as_dict()
        fws = wf_dict["fws"]

        print(fws)

        assert len(fws) == 5

        assert fws[-1]["name"] == "Bi-irrep"

    @pytest.mark.xfail  # Will fail if no local MongoDB connection
    def test_insert(self, bi):

        formula = bi.composition.reduced_formula
        irrep_out = IrrepOutput(os.path.join(test_dir, "Na3Bi_irrep.txt"))
        fw_spec = {
            "formula": formula,
            "efermi": 2.6723,
            "nelect": 26,
            "structure": bi.as_dict(),
        }

        db_file = os.path.join(db_dir, "db.json")
        toDb = IrrepToDb(irvsp_out=irrep_out, db_file=db_file)

        toDb.run_task(fw_spec)

        db = VaspCalcDb.from_db_file(db_file)
        db.collection = db.db["irvsp"]

        entry = db.collection.find_one({"formula": formula})

        assert entry["efermi"] == 2.6723


if __name__ == "__main__":
    pytest.main()
