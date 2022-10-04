import unittest
import numpy as np
from boRat import stress


class TestPCS(unittest.TestCase):

    def setUp(self) -> None:
        # self.s = stress.Stress()
        self.s = stress.Stress.from_PCS(SH=1, Sh=2, Sv=3)

    def test_diagonal(self):
        """Testing constructing diagonal stress matrix from principal stresses"""
        result = np.array([[1, 0, 0.], [0., 2, 0.], [0, 0, 3]])
        self.assertTrue(np.array_equal(self.s.stress, result))


class TestPCStoNEV(unittest.TestCase):

    def setUp(self) -> None:
        # self.s = stress.Stress()
        self.s = stress.Stress.from_PCS(SH=1, Sh=2, Sv=3)

    def test_SH_N(self):
        """SHazi - NS"""
        result = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        self.s.rot_PCS_to_NEV(SHazi=0)
        self.assertTrue(np.isclose(self.s.stress, result).all())

    def test_SH_E(self):
        """SHazi - EW"""
        result = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 3]])
        self.s.rot_PCS_to_NEV(SHazi=90)
        self.assertTrue(np.isclose(self.s.stress, result).all())

    def test_SH_S(self):
        """SHazi - SN = NS"""
        result = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        self.s.rot_PCS_to_NEV(SHazi=180)
        self.assertTrue(np.isclose(self.s.stress, result).all())

    def test_SH_W(self):
        """SHazi - WE = EW"""
        result = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 3]])
        self.s.rot_PCS_to_NEV(SHazi=270)
        self.assertTrue(np.isclose(self.s.stress, result).all())


class TestNEVtoTOH(unittest.TestCase):

    def setUp(self) -> None:
        # self.s = stress.Stress()
        self.s = stress.Stress.from_PCS(SH=1, Sh=2, Sv=3, SHazi=0)
        # self.s = self.nev.rot_PCS_to_NEV(SHazi=0)

    def test_horizontal_N(self):
        """SH N to horizontal N"""
        result = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
        rot = self.s.rot_NEV_to_TOH(hazi=0, hdev=90)
        self.assertTrue(np.isclose(rot.stress, result).all())

    def test_horizontal_E(self):
        """SH N to horizontal E"""
        result = np.array([[3, 0, 0], [0, 1, 0], [0, 0, 2]])
        rot = self.s.rot_NEV_to_TOH(hazi=90, hdev=90)
        self.assertTrue(np.isclose(rot.stress, result).all())

    def test_horizontal_S(self):
        """SH N to horizontal S"""
        result = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
        rot = self.s.rot_NEV_to_TOH(hazi=180, hdev=90)
        self.assertTrue(np.isclose(rot.stress, result).all())

    def test_horizontal_W(self):
        """SH N to horizontal W"""
        result = np.array([[3, 0, 0], [0, 1, 0], [0, 0, 2]])
        rot = self.s.rot_NEV_to_TOH(hazi=270, hdev=90)
        self.assertTrue(np.isclose(rot.stress, result).all())


class TestCart2cyl(unittest.TestCase):

    def setUp(self) -> None:
        # self.s = stress.Stress()
        self.s = stress.Stress.from_PCS(SH=1, Sh=2, Sv=3, SHazi=0)
        # self.s = self.nev.rot_PCS_to_NEV(SHazi=0)

    def test_theta_90(self):
        """Theta = 90 deg"""
        result = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 3]])
        cyl = self.s.cart2cyl(theta=90)
        self.assertTrue(np.isclose(cyl.stress, result).all())

    def test_theta_180(self):
        """Theta = 180 deg"""
        result = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        cyl = self.s.cart2cyl(theta=180)
        self.assertTrue(np.isclose(cyl.stress, result).all())


if __name__ == '__main__':
    unittest.main(verbosity=3)
