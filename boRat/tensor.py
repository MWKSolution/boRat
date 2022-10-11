import numpy as np
from boRat.config import tolerance as tol


class BondTransformationNotImplemented(Exception):
    pass


class TensorVoigt:
    """Base class for tensor in Voigt notation (6x6)"""
    def __init__(self, tensor=np.zeros((6, 6), dtype=np.float64)):
        self.tensor = tensor

    def rot(self, rotation):
        """Rotate tensor in Voigt notation using Bond transformation matrix calculated from given angles."""
        rotation_matrix = rotation.as_matrix()
        bt_matrix = self.bond_transformation(rotation_matrix)
        rotated = self.__class__()  # for inherited classes !!! bond trans. matrix is only aplicable for compliance or stiffness
        rotated.tensor = bt_matrix @ self.tensor @ bt_matrix.transpose()
        return rotated

    def rot_inplace(self, rotation):
        """Rotate tensor in Voigt notation using Bond transformation matrix calculated from given angles."""
        rotation_matrix = rotation.as_matrix()
        bt_matrix = self.bond_transformation(rotation_matrix)
        self.tensor = bt_matrix @ self.tensor @ bt_matrix.transpose()

    def __repr__(self):
        return f'{self.__class__.__name__}(\n{self.tensor!s})'

    def clean(self):
        """"Get rid of very small numbers in tensor"""
        self.tensor[np.abs(self.tensor) < tol] = 0.0

    @staticmethod
    def bond_transformation(a):
        """Function for creating Bond transformation matrix(6x6) from given 3x3 rotation matrix.
        !!! But only for compliance or stiffness !!!
        Defined here because there is reference in 'rot' method."""
        # B A Auld - Acoustic fields and waves in solids. Volume 1 - Interscience(1973). p. 73 - 76
        raise BondTransformationNotImplemented('Implemented only for child classes: Compliance, Stiffness.')


class Compliance(TensorVoigt):

    @staticmethod
    def bond_transformation(a):
        """Function for creating Bond transformation matrix(6x6) from given 3x3 rotation matrix for compliance"""
        lxx, lxy, lxz = a[0, 0], a[0, 1], a[0, 2]
        lyx, lyy, lyz = a[1, 0], a[1, 1], a[1, 2]
        lzx, lzy, lzz = a[2, 0], a[2, 1], a[2, 2]
        # rotation of compliance tensor for strains
        bond = np.array([[lxx ** 2, lxy ** 2, lxz ** 2, lxy * lxz, lxz * lxx, lxy * lxx],
                         [lyx ** 2, lyy ** 2, lyz ** 2, lyz * lyy, lyz * lyx, lyy * lyx],
                         [lzx ** 2, lzy ** 2, lzz ** 2, lzz * lzy, lzz * lzx, lzy * lzx],
                         [2 * lzx * lyx, 2 * lzy * lyy, 2 * lzz * lyz, lzz * lyy + lzy * lyz, lzz * lyx + lzx * lyz,
                          lzx * lyy + lzy * lyx],
                         [2 * lzx * lxx, 2 * lzy * lxy, 2 * lzz * lxz, lzz * lxy + lzy * lxz, lzz * lxx + lzx * lxz,
                          lzx * lxy + lzy * lxx],
                         [2 * lyx * lxx, 2 * lxy * lyy, 2 * lxz * lyz, lxz * lyy + lxy * lyz, lxz * lyx + lxx * lyz,
                          lxx * lyy + lxy * lyx]])
        return bond


class Stiffness(TensorVoigt):

    @staticmethod
    def bond_transformation(a):
        """Function for creating Bond transformation matrix(6x6) from given 3x3 rotation matrix for stiffness"""
        lxx, lxy, lxz = a[0, 0], a[0, 1], a[0, 2]
        lyx, lyy, lyz = a[1, 0], a[1, 1], a[1, 2]
        lzx, lzy, lzz = a[2, 0], a[2, 1], a[2, 2]
        # rotating of stiffness tensor for stresses
        bond = np.array([[lxx ** 2, lxy ** 2, lxz ** 2, 2 * lxy * lxz, 2 * lxz * lxx, 2 * lxy * lxx],
                         [lyx ** 2, lyy ** 2, lyz ** 2, 2 * lyz * lyy, 2 * lyz * lyx, 2 * lyy * lyx],
                         [lzx ** 2, lzy ** 2, lzz ** 2, 2 * lzz * lzy, 2 * lzz * lzx, 2 * lzy * lzx],
                         [lzx * lyx, lzy * lyy, lzz * lyz, lzz * lyy + lzy * lyz, lzz * lyx + lzx * lyz,
                          lzx * lyy + lzy * lyx],
                         [lzx * lxx, lzy * lxy, lzz * lxz, lzz * lxy + lzy * lxz, lzz * lxx + lzx * lxz,
                          lzx * lxy + lzy * lxx],
                         [lyx * lxx, lxy * lyy, lxz * lyz, lxz * lyy + lxy * lyz, lxz * lyx + lxx * lyz,
                          lxx * lyy + lxy * lyx]])
        return bond


if __name__ == '__main__':
    t = TensorVoigt()
    # t.bond_transformation('x')
