import numpy as np
from scipy.spatial.transform import Rotation as Rot
from boRat.config import tolerance as tol





class TensorVoigt:
    """Base class for tensor in Voigt notation (6x6)"""
    def __init__(self):
        self.tensor = np.zeros((6, 6), dtype=np.float64)

    def rot(self, mode, x=0, y=0, z=0):
        """Rotate tensor in Voigt notation using Bond transformation matrix calculated from given angles."""
        rotation = Rot.from_euler(mode, [x, y, z], degrees=True)
        rotation_matrix = rotation.as_matrix()
        bt_matrix = self.bond_transformation(rotation_matrix)
        rotated = TensorVoigt()
        rotated.tensor = bt_matrix @ self.tensor @ bt_matrix.transpose()
        return rotated

    def __repr__(self):
        return f'TensorVoigt(\n{self.tensor!s})'

    def clean(self):
        """"Get rid of very small numbers in tensor"""
        self.tensor[np.abs(self.tensor) < tol] = 0.0

    @staticmethod
    def bond_transformation(a):
        """Function for creating Bond transformation matrix(6x6) from given 3x3 rotation matrix."""
        # https://scicomp.stackexchange.com/questions/35600/4th-order-tensor-rotation-sources-to-refer
        return np.zeros((6, 6), dtype=np.float64)


class Compliance(TensorVoigt):

    @staticmethod
    def bond_transformation(a):
        """Function for creating Bond transformation matrix(6x6) from given 3x3 rotation matrix."""
        l11, l12, l13 = a[0, 0], a[0, 1], a[0, 2]
        l21, l22, l23 = a[1, 0], a[1, 1], a[1, 2]
        l31, l32, l33 = a[2, 0], a[2, 1], a[2, 2]
        # rotation of compliance tensor for rotation of strains
        bond = np.array([[l11 ** 2, l12 ** 2, l13 ** 2, l12 * l13, l13 * l11, l12 * l11],
                         [l21 ** 2, l22 ** 2, l23 ** 2, l23 * l22, l23 * l21, l22 * l21],
                         [l31 ** 2, l32 ** 2, l33 ** 2, l33 * l32, l33 * l31, l32 * l31],
                         [2 * l31 * l21, 2 * l32 * l22, 2 * l33 * l23, l33 * l22 + l32 * l23, l33 * l21 + l31 * l23,
                          l31 * l22 + l32 * l21],
                         [2 * l31 * l11, 2 * l32 * l12, 2 * l33 * l13, l33 * l12 + l32 * l13, l33 * l11 + l31 * l13,
                          l31 * l12 + l32 * l11],
                         [2 * l21 * l11, 2 * l12 * l22, 2 * l13 * l23, l13 * l22 + l12 * l23, l13 * l21 + l11 * l23,
                          l11 * l22 + l12 * l21]])

        return bond


class Stiffness(TensorVoigt):

    @staticmethod
    def bond_transformation(a):
        """Function for creating Bond transformation matrix(6x6) from given 3x3 rotation matrix."""
        l11, l12, l13 = a[0, 0], a[0, 1], a[0, 2]
        l21, l22, l23 = a[1, 0], a[1, 1], a[1, 2]
        l31, l32, l33 = a[2, 0], a[2, 1], a[2, 2]
        # rotating of stiffness tensor? for rotation of stresses
        bond = np.array([[l11 ** 2, l12 ** 2, l13 ** 2, 2 * l12 * l13, 2 * l13 * l11, 2 * l12 * l11],
                          [l21 ** 2, l22 ** 2, l23 ** 2, 2 * l23 * l22, 2 * l23 * l21, 2 * l22 * l21],
                          [l31 ** 2, l32 ** 2, l33 ** 2, l33 * l32, l33 * l31, l32 * l31],
                          [l31 * l21, l32 * l22, l33 * l23, l33 * l22 + l32 * l23, l33 * l21 + l31 * l23,
                           l31 * l22 + l32 * l21],
                          [l31 * l11, l32 * l12, l33 * l13, l33 * l12 + l32 * l13, l33 * l11 + l31 * l13,
                           l31 * l12 + l32 * l11],
                          [l21 * l11, l12 * l22, l13 * l23, l13 * l22 + l12 * l23, l13 * l21 + l11 * l23,
                           l11 * l22 + l12 * l21]])

        return bond
