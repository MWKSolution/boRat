import numpy as np
from scipy.spatial.transform import Rotation as Rot
from boRat.config import intrinsic, extrinsic


class WellboreOrientation:
    """Class defining wellbore orientation: azimuth and deviation along with unit vector parallel to wellbore axis."""
    def __init__(self, hazi=0, hdev=0, Pw=0):
        self.hazi = hazi  # hole azimuth
        self.hdev = hdev  # hole deviation
        self.Pw = Pw  # mud pressure
        self.vector = self.get_parallel_vector()

    def get_parallel_vector(self):
        """Get vector which is parallel to borehole axis by rotating unit vector perpendicular to earth surface."""
        init = np.array([0, 0, 1])  # vector pointing down!!!
        rotation = Rot.from_euler(extrinsic, [0, self.hdev, self.hazi], degrees=True)
        return rotation.apply(init)

    def __repr__(self):
        return f'WellboreOrientation(Hole azimuth: {self.hazi!s}, Hole deviation: {self.hdev!s}, Parallel vector: {self.vector!s} NEV)'
