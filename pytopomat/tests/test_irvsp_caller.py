import warnings
import os
import pytest
import pandas as pd

from monty.os.path import which

from pytopomat.irvsp_caller import IRVSPCaller, IRVSPOutput

test_dir = os.path.join(os.path.dirname(__file__), "../..", "test_files")
IRVSPEXE = which("irvsp")


@pytest.fixture
def ic():
    """Returns an IRVSPCaller instance in the current directory."""

    return IRVSPCaller(".")


@pytest.fixture
def parity_eigenvals():
    """Returns IRVSPOutput instance with CrO2 data."""
    out = IRVSPOutput(os.path.join(test_dir, "CrO2_outir.txt"))

    return out.parity_eigenvals


def test_output_trims(parity_eigenvals):

    assert len(parity_eigenvals) == 8


def test_output_spins(parity_eigenvals):

    assert len(parity_eigenvals["gamma"].keys()) == 2


def test_parsing(parity_eigenvals):

    assert parity_eigenvals["gamma"]["down"]["inversion_eigenval"][0] == 1.0


if __name__ == "__main__":
    pytest.main()
