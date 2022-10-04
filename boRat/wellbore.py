import numpy as np
from scipy.spatial.transform import Rotation as Rot
from collections import namedtuple

Orientation = namedtuple('Orientation', ['hazi', 'hdev'])


class Wellbore:
    """Class defining wellbore orientation: azimuth and deviation along with unit vector parallel to wellbore axis."""
    def __init__(self, hazi=0, hdev=0, Pw=0):
        self.orien = Orientation(hazi=hazi, hdev=hdev)  # WB orientation - azimuth and deviation
        self.Pw = Pw  # mud pressure
        self.vector = self.get_parallel_vector_NEV()  # vector in NEV coordiantes

    def get_parallel_vector_NEV(self):
        """Get vector which is parallel to borehole axis by rotating unit vector perpendicular to earth surface."""
        init = np.array([0, 0, 1], dtype=np.float64)  # vector pointing down - parallel to vertical well !!!
        rotation = Rot.from_euler('ZY', [self.orien.hazi, self.orien.hdev], degrees=True)
        return rotation.apply(init)

    def __repr__(self):
        return f'WellboreOrientation({self.orien.hazi!s}, {self.orien.hdev!s}, {self.Pw!s})'

    def __str__(self):
        return f'WellboreOrientation(Hole azimuth: {self.orien.hazi!s}, Hole deviation: {self.orien.hdev!s}, Parallel vector: {self.vector!s} NEV, Mud pressure: {self.Pw!s})'


if __name__ == '__main__':
    wbo = Wellbore(hazi=45, hdev=45, Pw=10)
    print(f'str : {wbo!s}')
    print(f'repr: {wbo!r}')
