import warnings

import os
import unittest
import pandas as pd

from pytopomat.analyzer import Vasp2TraceCaller, Vasp2TraceOutput, BandParity

test_dir = os.path.join(os.path.dirname(__file__), "..", "test_files")


class BandParityTest(unittest.TestCase):

    pass


if __name__ == "__main__":
    unittest.main()
