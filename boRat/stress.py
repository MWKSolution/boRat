import numpy as np
from scipy.spatial.transform import Rotation as Rot
from boRat.config import tolerance as tol
from boRat.config import intrinsic, extrinsic


class Stress:
    """Base class for 3x3 stress tensor (matrix)"""
    def __init__(self):
        #  stress tensor
        self.stress = np.zeros((3, 3), dtype=np.float64)
        self.SHAzi = None

    def rot(self, mode, x=0, y=0, z=0):
        """Rotate 3x3 stress tensor using given angles.
        Returns new out-of-place Stress instance"""
        # get new instance of Stress (out-of-place)
        rotated = Stress()
        # get rotation matrix
        rotation = Rot.from_euler(mode, [x, y, z], degrees=True)
        rotation_matrix = rotation.as_matrix()
        # rotate
        rotated.stress = rotation_matrix @ self.stress @ rotation_matrix.transpose()
        return rotated

    def set_from_PCS(self, SH=0, Sh=0, Sz=0, SHAzi=0):
        """Get diagonal matrix from given 3 principal stresses.
        SH is along X axis, Sh - Y axis, Sz - Z axis.
        X is in the North direction, Y is in East direction,3 Z is down."""
        self.stress = np.zeros((3, 3))
        #  diagonal stresses
        self.stress[0, 0] = SH  # X
        self.stress[1, 1] = Sh  # Y
        self.stress[2, 2] = Sz  # Z
        self.SHAzi = SHAzi

    def cart2cyl(self, theta=0):
        """Transform 3x3 stress tensor from cartesian to cylindrical coordinates for given theta coordinate.
        Theta is angle between x and r unit vectors."""
        trans = self.rot(extrinsic, z=theta)
        return trans

    def clean(self):
        """"Get rid of very small numbers in tensor"""
        self.stress[np.abs(self.stress) < tol] = 0.0

    def __repr__(self):
        return f'Stress(\n{self.stress!s})'



