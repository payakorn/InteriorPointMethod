import unittest
from sparse_interior import load_data_mps_matlab
import numpy as np


class TestDataDimension(unittest.TestCase):

    c, Aineq, bineq, Aeq, beq, lb, ub = load_data_mps_matlab("ADLITTLE")
    row_Aineq, col_Aineq = np.shape(Aineq)
    row_bineq, col_bineq = np.shape(bineq)
    row_Aeq, col_Aeq = np.shape(Aeq)
    row_beq, col_beq = np.shape(beq)

    def test_ow_inequality(self):
        self.assertEqual(self.row_Aineq, self.row_bineq)

    def test_row_quality(self):
        self.assertEqual(self.row_Aeq, self.row_beq)

    def test_col_bineq(self):
        self.assertEqual(self.col_bineq, 1)

    def test_col_beq(self):
        self.assertEqual(self.col_beq, 1)


if __name__ == "__main__":
    unittest.main()

