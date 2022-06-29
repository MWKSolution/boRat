import unittest
import numpy as np
from boRat.rock import ISORock, TIVRock, ORTRock
from boRat.config import intrinsic, extrinsic

iso = ISORock().compliance
tiv = TIVRock().compliance
ort = ORTRock().compliance


class TestTensorRotation(unittest.TestCase):

    def test_rotISO(self):
        """Tensor for ISOtropic rock is not changed by any rotation"""
        rotated = iso.rot(intrinsic, x=25, y=56, z=170).tensor
        self.assertTrue(np.isclose(rotated, iso.tensor).all())
        rotated = iso.rot(intrinsic, x=200, y=100, z=70).tensor
        self.assertTrue(np.isclose(rotated, iso.tensor).all())
        rotated = iso.rot(intrinsic, x=10, y=20, z=30).tensor
        self.assertTrue(np.isclose(rotated, iso.tensor).all())

    def test_rotTIV(self):
        """Tensor for TIV rock is not changed by rotation around Z axis (if Z axis is parallel to symmetry axis !!!),
        or by putting it upside down. Any other rotation gives different tensor"""
        rotated = tiv.rot(intrinsic, x=0, y=0, z=75).tensor
        self.assertTrue(np.isclose(rotated, tiv.tensor).all())
        rotated = tiv.rot(intrinsic, x=180, y=180, z=15).tensor
        self.assertTrue(np.isclose(rotated, tiv.tensor).all())
        rotated = tiv.rot(intrinsic, x=10, y=20, z=30).tensor
        self.assertFalse(np.isclose(rotated, tiv.tensor).all())

    def test_rotORT(self):
        """Tensor for TIV rock is not changed by rotation around Z axis (if Z axis is parallel to symmetry axis !!!),
        or by putting it upside down. Any other rotation gives different tensor"""
        rotated = ort.rot(intrinsic, x=0, y=0, z=75).tensor
        self.assertFalse(np.isclose(rotated, ort.tensor).all())
        rotated = ort.rot(intrinsic, x=0, y=0, z=180).tensor
        self.assertTrue(np.isclose(rotated, ort.tensor).all())
        rotated = ort.rot(intrinsic, x=0, y=180, z=0).tensor
        self.assertTrue(np.isclose(rotated, ort.tensor).all())
        rotated = ort.rot(intrinsic, x=180, y=0, z=0).tensor
        self.assertTrue(np.isclose(rotated, ort.tensor).all())
        rotated = ort.rot(intrinsic, x=10, y=20, z=30).tensor
        self.assertFalse(np.isclose(rotated, ort.tensor).all())

    def test_rot_back(self):
        """Tensor for TIV rock is not changed by rotation around Z axis (if Z axis is parallel to symmetry axis !!!),
        or by putting it upside down. Any other rotation gives different tensor"""
        rotated1 = ort.rot(intrinsic, x=10, y=20, z=30)
        rotated2 = rotated1.rot(extrinsic, x=-10, y=-20, z=-30).tensor
        self.assertTrue(np.isclose(rotated2, ort.tensor).all())


class TestBondMatrix(unittest.TestCase):

    @unittest.skip
    def test_bond_matrix(self):
        """Constructing bond matrix is tested when tensor rotation is tested."""
        pass


if __name__ == '__main__':
    unittest.main(verbosity=3)
