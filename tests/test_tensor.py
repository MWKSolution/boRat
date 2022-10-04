import unittest
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from boRat.rock import Rock


class TestTensorRotationCompliance(unittest.TestCase):

    def setUp(self) -> None:
        self.iso = Rock.ISO_from_moduli().compliance
        self.tiv = Rock.TIV_from_moduli().compliance
        # ort = Rock().compliance)

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

    # def test_rotORT(self):
    #     """Tensor for TIV rock is not changed by rotation around Z axis (if Z axis is parallel to symmetry axis !!!),
    #     or by putting it upside down. Any other rotation gives different tensor"""
    #     rotated = ort.rot(intrinsic, x=0, y=0, z=75).tensor
    #     self.assertFalse(np.isclose(rotated, ort.tensor).all())
    #     rotated = ort.rot(intrinsic, x=0, y=0, z=180).tensor
    #     self.assertTrue(np.isclose(rotated, ort.tensor).all())
    #     rotated = ort.rot(intrinsic, x=0, y=180, z=0).tensor
    #     self.assertTrue(np.isclose(rotated, ort.tensor).all())
    #     rotated = ort.rot(intrinsic, x=180, y=0, z=0).tensor
    #     self.assertTrue(np.isclose(rotated, ort.tensor).all())
    #     rotated = ort.rot(intrinsic, x=10, y=20, z=30).tensor
    #     self.assertFalse(np.isclose(rotated, ort.tensor).all())

    # def test_rot_back(self):
    #     """Tensor for TIV rock is not changed by rotation around Z axis (if Z axis is parallel to symmetry axis !!!),
    #     or by putting it upside down. Any other rotation gives different tensor"""
    #     rotated1 = ort.rot(intrinsic, x=10, y=20, z=30)
    #     rotated2 = rotated1.rot(extrinsic, x=-10, y=-20, z=-30).tensor
    #     self.assertTrue(np.isclose(rotated2, ort.tensor).all())


class TestTensorRotationStiffness(TestTensorRotationCompliance):
    """For rotating stiffness bond transformation matrix is a little different"""

    def setUp(self) -> None:
        self.iso = Rock.ISO_from_moduli().stiffness
        self.tiv = Rock.TIV_from_moduli().stiffness
        # ort = Rock().compliance)


class TestBondMatrix(unittest.TestCase):

    def test_bond_matrix(self):
        """Constructing bond matrix is tested when tensor rotation is tested."""
        print('Constructing bond matrix is tested when tensor rotation is tested.')
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=3)
