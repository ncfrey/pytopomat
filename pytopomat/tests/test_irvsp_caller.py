import warnings
import os
import pytest

from monty.os.path import which
from monty.serialization import dumpfn, loadfn

from pytopomat.irvsp_caller import IRVSPCaller, IRVSPOutput

test_dir = os.path.join(os.path.dirname(__file__), "../..", "test_files")
IRVSPEXE = which("irvsp")


class TestIrvsp(object):
    @pytest.fixture
    def ic(self):
        """Returns an IRVSPCaller instance in the current directory."""

        cwd = os.getcwd()
        return IRVSPCaller(cwd)

    @pytest.fixture
    def parity_eigenvals(self):
        """Returns IRVSPOutput parity eigenvalues for Bi2Se3 data."""
        out = IRVSPOutput(os.path.join(test_dir, "Bi2Se3_outir.txt"))

        return out.parity_eigenvals

    def test_output_save(self):
        out = IRVSPOutput(os.path.join(test_dir, "Bi2Se3_outir.txt"))
        out_dict = out.as_dict()
        out_from_dict = IRVSPOutput.from_dict(out_dict)

        dumpfn(out_from_dict, "tmp.json")
        out = loadfn("tmp.json")
        os.remove("tmp.json")

        assert out.soc == True

    def test_output_trims(self, parity_eigenvals):

        assert len(parity_eigenvals) == 8

    def test_output_bands(self, parity_eigenvals):

        assert len(parity_eigenvals["gamma"]["band_eigenval"]) == 32

    def test_parsing(self, parity_eigenvals):

        assert parity_eigenvals["gamma"]["inversion_eigenval"][0] == 2.0

    @pytest.fixture
    def spin_parity_eigenvals(self):
        """Returns IRVSPOutput parity eigenvalues for CrO2 data."""
        out = IRVSPOutput(os.path.join(test_dir, "CrO2_outir.txt"))

        return out.parity_eigenvals

    def test_spin_output_trims(self, spin_parity_eigenvals):

        assert len(spin_parity_eigenvals) == 8

    def test_output_spins(self, spin_parity_eigenvals):

        assert len(spin_parity_eigenvals["gamma"].keys()) == 2

    def test_spin_parsing(self, spin_parity_eigenvals):

        assert spin_parity_eigenvals["gamma"]["down"]["inversion_eigenval"][0] == 1.0


if __name__ == "__main__":
    pytest.main()
