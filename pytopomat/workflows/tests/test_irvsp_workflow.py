import warnings
import os
import pytest

from pymatgen import Structure, Lattice

from fireworks import LaunchPad
from atomate.utils.testing import AtomateTest, DB_DIR
from atomate.vasp.database import VaspCalcDb

from pytopomat.irvsp_caller import IRVSPOutput
from pytopomat.workflows.core import wf_irvsp
from pytopomat.workflows.fireworks import IrvspFW, StandardizeFW
from pytopomat.workflows.firetasks import RunIRVSP, IRVSPToDb, StandardizeCell

module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
db_dir = os.path.join(module_dir, "..", "common", "test_files")
test_dir = os.path.join(module_dir, "..", "..", "..", "test_files")


class TestIrvspWorkflow(object):
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

    def test_get_wflow(self, bi):

        wf = wf_irvsp(bi, magnetic=False, soc=True, v2t=True)
        wf_dict = wf.as_dict()
        fws = wf_dict["fws"]

        assert len(fws) == 6

        assert fws[-1]["name"] == "Bi-vasp2trace"

        assert fws[-2]["name"] == "Bi-irvsp"

    @pytest.mark.xfail  # Will fail if no local MongoDB connection
    def test_insert(self, bi):

        formula = bi.composition.reduced_formula
        irvsp_out = IRVSPOutput(os.path.join(test_dir, "Bi2Se3_outir.txt"))
        fw_spec = {"formula": formula, "efermi": 3.0, "structure": bi.as_dict()}

        db_file = os.path.join(db_dir, "db.json")
        toDb = IRVSPToDb(irvsp_out=irvsp_out, db_file=db_file)

        toDb.run_task(fw_spec)

        db = VaspCalcDb.from_db_file(db_file)
        db.collection = db.db["irvsp"]

        entry = db.collection.find_one({"formula": formula})

        assert entry["efermi"] == 3.0


if __name__ == "__main__":
    pytest.main()
