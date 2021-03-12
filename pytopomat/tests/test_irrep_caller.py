import os
import pytest

from monty.os.path import which
from monty.serialization import dumpfn, loadfn

from pytopomat.irrep_caller import IrrepCaller, IrrepOutput

test_dir = os.path.join(os.path.dirname(__file__), "../..", "test_files")
IRREPEXE = which("irrep")

# TODO: add spin-polarized test for irrep


class TestIrrep(object):
    @pytest.fixture
    def ic(self):
        """Returns an IrrepCaller instance in the current directory."""

        cwd = os.getcwd()
        return IrrepCaller(cwd)

    @pytest.fixture
    def parity_eigenvals(self):
        """Returns IrrepOutput parity eigenvalues for Na3Bi data."""
        out = IrrepOutput(os.path.join(test_dir, "Na3Bi_irrep.txt"))

        return out.parity_eigenvals

    def test_output_save(self):
        out = IrrepOutput(os.path.join(test_dir, "Na3Bi_irrep.txt"))
        out_dict = out.as_dict()
        out_from_dict = IrrepOutput.from_dict(out_dict)

        dumpfn(out_from_dict, "tmp.json")
        out = loadfn("tmp.json")
        os.remove("tmp.json")

        assert out.efermi is None
        assert out.saved_bands == 128
        assert out.starting_band == 1
        assert out.energy_cutoff == 520.0
        assert out.spacegroup_no == 225
        assert out.spin_polarized is None

    def test_output_trims(self, parity_eigenvals):

        assert len(parity_eigenvals) == 8

    def test_output_bands(self, parity_eigenvals):

        assert len(parity_eigenvals["gamma"]["band_eigenval"]) == 43

    def test_parsing(self, parity_eigenvals):

        assert parity_eigenvals["gamma"]["inversion_eigenval"][10] == 4.0


if __name__ == "__main__":
    pytest.main()
