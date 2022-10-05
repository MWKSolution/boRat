import unittest
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from boRat.rock import Rock


class TestTensorRotationCompliance(unittest.TestCase):

    def setUp(self) -> None:
        self.iso = Rock.ISO_from_moduli().compliance
        self.tiv = Rock.TIV_from_moduli().compliance
        self.ort = Rock.ORT_from_moduli().compliance

    def test_rotISO(self):
        """Tensor for ISOtropic rock is not changed by any rotation"""
        rotation = Rot.from_euler('XYZ', [25, 56, 170], degrees=True)
        rotated = self.iso.rot(rotation).tensor
        self.assertTrue(np.isclose(rotated, self.iso.tensor).all())

        rotation = Rot.from_euler('XYZ', [200, 100, 70], degrees=True)
        rotated = self.iso.rot(rotation).tensor
        self.assertTrue(np.isclose(rotated, self.iso.tensor).all())

        rotation = Rot.from_euler('XYZ', [10, 20, 30], degrees=True)
        rotated = self.iso.rot(rotation).tensor
        self.assertTrue(np.isclose(rotated, self.iso.tensor).all())

    def test_rotTIV(self):
        """Tensor for TIV rock is not changed by rotation around Z axis (if Z axis is parallel to symmetry axis !!!),
        or by putting it upside down. Any other rotation gives different tensor"""
        rotation = Rot.from_euler('XYZ', [0, 0, 75], degrees=True)
        rotated = self.tiv.rot(rotation).tensor
        self.assertTrue(np.isclose(rotated, self.tiv.tensor).all())

        rotation = Rot.from_euler('XYZ', [180, 0, 15], degrees=True)
        rotated = self.tiv.rot(rotation).tensor
        self.assertTrue(np.isclose(rotated, self.tiv.tensor).all())

        rotation = Rot.from_euler('XYZ', [0, 180, 37], degrees=True)
        rotated = self.tiv.rot(rotation).tensor
        self.assertTrue(np.isclose(rotated, self.tiv.tensor).all())

        rotation = Rot.from_euler('XYZ', [10, 20, 30], degrees=True)
        rotated = self.tiv.rot(rotation).tensor
        self.assertFalse(np.isclose(rotated, self.tiv.tensor).all())

    def test_rotORT(self):
        """Tensor for ORT rock is not changed by rotation 180 degs around any axis (mirror reflections are the same)."""
        rotation = Rot.from_euler('XYZ', [0, 0, 75], degrees=True)
        rotated = self.ort.rot(rotation).tensor
        self.assertFalse(np.isclose(rotated, self.ort.tensor).all())

        rotation = Rot.from_euler('XYZ', [180, 0, 0], degrees=True)
        rotated = self.ort.rot(rotation).tensor
        self.assertTrue(np.isclose(rotated, self.ort.tensor).all())

        rotation = Rot.from_euler('XYZ', [0, 180, 0], degrees=True)
        rotated = self.ort.rot(rotation).tensor
        self.assertTrue(np.isclose(rotated, self.ort.tensor).all())

        rotation = Rot.from_euler('XYZ', [0, 0, 180], degrees=True)
        rotated = self.ort.rot(rotation).tensor
        self.assertTrue(np.isclose(rotated, self.ort.tensor).all())

        rotation = Rot.from_euler('XYZ', [10, 20, 30], degrees=True)
        rotated = self.ort.rot(rotation).tensor
        self.assertFalse(np.isclose(rotated, self.ort.tensor).all())

    def test_rot_back(self):
        """Just test rotating tensor forth and back"""
        rotation1 = Rot.from_euler('XYZ', [10, 20, 30], degrees=True)
        rotated1 = self.ort.rot(rotation1)
        rotation2 = Rot.from_euler('ZYX', [-30, -20, -10], degrees=True)
        rotated2 = rotated1.rot(rotation2).tensor
        self.assertTrue(np.isclose(rotated2, self.ort.tensor).all())


class TestTensorRotationStiffness(TestTensorRotationCompliance):
    """For rotating stiffness - bond transformation matrix is a little different"""

    def setUp(self) -> None:
        self.iso = Rock.ISO_from_moduli().stiffness
        self.tiv = Rock.TIV_from_moduli().stiffness
        self.ort = Rock.ORT_from_moduli().stiffness


class TestBondMatrix(unittest.TestCase):

    def test_bond_matrix(self):
        """Constructing bond matrix is tested when tensor rotation is tested."""
        print('Constructing bond matrix is tested when tensor rotation is tested.')
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=3)
