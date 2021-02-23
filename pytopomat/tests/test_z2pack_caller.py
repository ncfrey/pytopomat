import os
import pytest

from monty.serialization import dumpfn, loadfn
import z2pack

from pytopomat.z2pack_caller import Z2Output

test_dir = os.path.join(os.path.dirname(__file__), "../../test_files/")


class TestZ2Pack(object):
    @pytest.fixture
    def z2out(self):
        """Returns Z2Output instance with Bi kx=0 surface data."""
        result = z2pack.io.load(test_dir + "res_1.json")
        out = Z2Output(result, "kx_0")

        return out

    def test_output_save(self, z2out):

        z2o_dict = z2out.as_dict()
        z2o_from_dict = Z2Output.from_dict(z2o_dict)
        dumpfn(z2o_from_dict, "tmp.json")
        out = loadfn("tmp.json")
        os.remove("tmp.json")

        assert out.z2_invariant == 0


if __name__ == "__main__":
    pytest.main()
