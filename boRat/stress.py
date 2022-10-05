import numpy as np
from scipy.spatial.transform import Rotation as Rot
from boRat.config import tolerance as tol


class Stress:
    """Base class for 3x3 stress tensor (matrix)"""
    def __init__(self, stress=np.zeros((3, 3), dtype=np.float64), SHazi=0.0):
        #  stress tensor
        self.stress = stress
        self.rot_PCS_to_NEV(SHazi=SHazi)

    @classmethod
    def from_PCS(cls, SH=0.0, Sh=0.0, Sv=0.0, SHazi=0.0):
        """Get diagonal matrix from given 3 principal stresses.
        SH is along X axis, Sh - Y axis, Sz - Z axis.
        X is in the North direction, Y is in East direction,3 Z is down.
        SHAzi gives azimuth of X axis."""
        _stress = np.diag(np.array([SH, Sh, Sv], dtype=np.float64))
        return cls(_stress, SHazi)

    def rot_PCS_to_NEV(self, SHazi=0.0):  # only SH azimuth, Sz always vertical !!!
        rotation = Rot.from_euler('YZ', [0, SHazi], degrees=True)
        self.rot_inplace(rotation)

    def rot_NEV_to_TOH(self, hazi=0, hdev=0):
        rotation = Rot.from_euler('YZ', [-hdev, -hazi], degrees=True)
        return self.rot(rotation)

    def rot(self, rotation=None):
        """Rotate 3x3 stress tensor using given angles.
        Returns new out-of-place Stress instance"""
        # get new instance of Stress (out-of-place)
        rotated = Stress()
        rotation_matrix = rotation.as_matrix()
        rotated.stress = rotation_matrix @ self.stress @ rotation_matrix.transpose()
        return rotated

    def rot_inplace(self, rotation=None):
        """Rotate 3x3 stress tensor using given angles.
        Returns new out-of-place Stress instance"""
        rotation_matrix = rotation.as_matrix()
        self.stress = rotation_matrix @ self.stress @ rotation_matrix.transpose()

    def cart2cyl(self, theta=0):
        """Transform 3x3 stress tensor from cartesian to cylindrical coordinates for given theta coordinate.
        Theta is angle between x and r unit vectors."""
        rotation = Rot.from_euler('XYZ', [0, 0, -theta], degrees=True)
        return self.rot(rotation)

    def clean(self):
        """"Get rid of very small numbers in tensor"""
        self.stress[np.abs(self.stress) < tol] = 0.0

    def __repr__(self):
        return f'Stress(\n{self.stress!s})'


if __name__ == '__main__':
    s0 = Stress()
    print(s0)
    s1 = Stress.from_PCS(1, 2, 3)
    print(s1)
    s2 = Stress.from_PCS(1, 2, 3, 45)
    print(s2)




