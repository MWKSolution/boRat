import unittest
import numpy as np
from boRat import stress
from boRat.config import intrinsic, extrinsic


class TestPCS(unittest.TestCase):

    def setUp(self) -> None:
        self.s = stress.Stress()
        self.s.set_from_PCS(SH=1, Sh=2, Sz=3)

    def test_diagonal(self):
        """Testing constructing diagonal stress matrix from principal stresses"""
        result = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        self.assertTrue(np.array_equal(self.s.stress, result))


class TestStressRotation(unittest.TestCase):

    def setUp(self) -> None:
        self.s = stress.Stress()
        self.s.set_from_PCS(SH=1, Sh=2, Sz=3)

    def test_rot_x_90(self):
        """Rotation 90 deg around x axis"""
        result = np.array([[1, 0, 0], [0, 3, 0], [0, 0, 2]])
        rot = self.s.rot(extrinsic, x=90, y=0, z=0)
        self.assertTrue(np.isclose(rot.stress, result).all())

    def test_rot_y_90(self):
        """Rotation 90 deg around y axis"""
        result = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
        rot = self.s.rot(extrinsic, x=0, y=90, z=0)
        self.assertTrue(np.isclose(rot.stress, result).all())

    def test_rot_z_90(self):
        """Rotation 90 deg around z axis"""
        result = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 3]])
        rot = self.s.rot(extrinsic, x=0, y=0, z=90)
        self.assertTrue(np.isclose(rot.stress, result).all())

    def test_rot_xyz_90_90_90(self):
        """Rotation 90 deg around all axes"""
        result = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
        rot = self.s.rot(extrinsic, x=90, y=90, z=90)
        self.assertTrue(np.isclose(rot.stress, result).all())

    def test_rot_back(self):
        result = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        rot1 = self.s.rot(intrinsic, x=10, y=20, z=30)
        rot2 = rot1.rot(extrinsic, x=-10, y=-20, z=-30)
        self.assertTrue(np.isclose(rot2.stress, result).all())


class TestCart2cyl(unittest.TestCase):

    def setUp(self) -> None:
        self.s = stress.Stress()
        self.s.set_from_PCS(SH=1, Sh=2, Sz=3)

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
