import warnings
import os
import numpy as np
import pytest

from pytopomat.irvsp_caller import IRVSPOutput
from pytopomat.analyzer import BandParity

test_dir = os.path.join(os.path.dirname(__file__), "../../test_files/")

class TestIrvsp(object):
    @pytest.fixture
    def bp(self):
        """Returns BandParity instance on test data."""

        irvsp_out = IRVSPOutput(test_dir + "CrO2_outir.txt")
        band_parity = BandParity(irvsp_out, spin_polarized=True, efermi=0.0)

        return band_parity

    def test_compute_z2(self, bp):
    	z2 = bp.compute_z2(tol=3)
    	tz2 = np.array([1., 0., 0., 0.])

    	np.testing.assert_array_equal(z2, tz2)

    def test_compute_z4(self, bp):
    	z4 = bp.compute_z4()

    	assert z4 == 1.0


if __name__ == "__main__":
    pytest.main()