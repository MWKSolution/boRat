import unittest
import numpy as np
from boRat.wellbore import WellboreOrientation


class TestWellboreVector(unittest.TestCase):

    def setUp(self) -> None:
        self.vertical = np.array([0, 0, 1])
        self.horizontal_N = np.array([1, 0, 0])
        self.horizontal_E = np.array([0, 1, 0])
        self.horizontal_S = np.array([-1, 0, 0])
        self.horizontal_W = np.array([0, -1, 0])

    def test_vertical(self):
        """Vertical well (drilled down!)"""
        wbo = WellboreOrientation().vector
        self.assertTrue(np.isclose(wbo, self.vertical).all())

    def test_horizontal_N(self):
        """Horizontal well drilled to the North"""
        wbo = WellboreOrientation(hdev=90).vector
        self.assertTrue(np.isclose(wbo, self.horizontal_N).all())

    def test_horizontal_E(self):
        """Horizontal well drilled to the East"""
        wbo = WellboreOrientation(hazi=90, hdev=90).vector
        self.assertTrue(np.isclose(wbo, self.horizontal_E).all())

    def test_horizontal_S(self):
        """Horizontal well drilled to the South"""
        wbo = WellboreOrientation(hazi=180, hdev=90).vector
        self.assertTrue(np.isclose(wbo, self.horizontal_S).all())

    def test_horizontal_W(self):
        """Horizontal well drilled to the West"""
        wbo = WellboreOrientation(hazi=270, hdev=90).vector
        self.assertTrue(np.isclose(wbo, self.horizontal_W).all())


if __name__ == '__main__':
    unittest.main(verbosity=3)
