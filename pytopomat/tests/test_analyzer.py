import os
import numpy as np
import pytest

from pytopomat.irrep_caller import IrrepOutput
from pytopomat.analyzer import BandParity

test_dir = os.path.join(os.path.dirname(__file__), "../../test_files/")


class TestIrvsp(object):
    @pytest.fixture
    def bp(self):
        """Returns BandParity instance on test data."""

        irvsp_out = IrrepOutput(test_dir + "Na3Bi_irrep.txt")
        band_parity = BandParity(irvsp_out, efermi=2.6723, nelect=26)

        return band_parity

    @pytest.fixture
    def bp_sp(self):
        """Returns BandParity instance on test data (spin-polarized)."""

        irvsp_out = IrrepOutput(test_dir + "CrO2_sp_irrep.txt")
        band_parity = BandParity(
            irvsp_out, spin_polarized=True, efermi=4.6476, nelect=48
        )

        return band_parity

    def test_compute_z2(self, bp):
        z2 = bp.compute_z2(tol=-1)
        tz2 = np.array([1.0, 0.0, 0.0, 0.0])

        np.testing.assert_array_equal(z2, tz2)

    def test_compute_z4(self, bp_sp):
        z4 = bp_sp.compute_z4()

        assert z4 == 3.0

    def test_screen_magnetic_parity(self, bp_sp):
        screening = bp_sp.screen_magnetic_parity()

        assert screening == {
            "insulator": False,
            "semimetal_candidate": True,
            "polarization_bqhc": False,
            "magnetoelectric": False,
        }


if __name__ == "__main__":
    pytest.main()
