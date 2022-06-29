import numpy as np
from boRat.tensor import Compliance, Stiffness
from scipy.spatial.transform import Rotation as Rot
from boRat.config import __log__
from boRat.config import intrinsic, extrinsic

# :todo: get typical rocks...
ISO_ROCK = dict(E=30.14, PR=0.079)
TIV_ROCK = dict(Ev=15.42, Eh=31.17, PRv=0.32, PRhh=0.079, Gv=7.05)
ORT_ROCK = dict(Ex=30, Ey=20, Ez=15, PRyx=0.3, PRzx=0.2, PRzy=0.1, Gyz=6, Gxz=7, Gxy=8)


class FormationDip:
    """Class defining formation dip: dip and direction along with vector perpendicular to bedding.
     For flat bedding vector is pointing in Z axis direction (down!!!)."""
    def __init__(self, dip=0, dir=0):
        self.dip = dip  # formation dip
        self.dir = dir  # dip direction
        self.vector = self.get_normal_vector()

    def get_normal_vector(self):
        """Vector perpendicular to bedding. Since for flat bedding it is pointing down (Z axis direction) it has to be rotated with -dip value!"""
        init = np.array([0, 0, 1])
        rotation = Rot.from_euler(extrinsic, [0, -self.dip, self.dir], degrees=True)
        rotation_matrix = rotation.as_matrix()
        return rotation.apply(init)

    def __repr__(self):
        return f'FormationDip(Dip angle: {self.dip!s}, Dip direction: {self.dir!s}, Normal vector: {self.vector!s} NEV)'


class Rock:
    """Class describing rock mechanical properties:
        compliance - compliance tensor in Voigt notation,
        stiffness - stiffness tensor in Voigt notation,
        ..."""
    # :todo: lampiere verification and other verifications for E, PR and G
    def __init__(self, **kwargs):
        self.compliance = Compliance()
        self.stiffness = Stiffness()
        self.symmetry = None

    def set_compliance(self, *kwargs):
        """Set compliance tensor for the rock basing on given elastic moduli of the rock.
        Function implementation depends on level of anisotropy of the rock and is defined in child classes of Rock:
        ISO, TIV, ORT."""
        pass

    def get_stiffness(self):
        """Get stiffness from compliance. By definition, it is inversion of compliance."""
        self.stiffness.tensor = np.linalg.inv(self.compliance.tensor)

    def rot(self, x=0, y=0, z=0):
        """Rotate compliance tensor with given angles and get stiffness tensor."""
        rotated = Rock()
        rotated.compliance = self.compliance.rot(x, y, z)
        rotated.get_stiffness()
        rotated.symmetry = self.symmetry
        return rotated

    def clean(self):
        self.compliance.clean()
        self.stiffness.clean()

    def __repr__(self):
        return f'Rock:(\nCompliance:({self.compliance!s})\nStiffness:({self.stiffness!s}))'


class ISORock(Rock):
    """Class for isotropic (ISO) rocks."""
    def __init__(self,
                 E=ISO_ROCK['E'],
                 PR=ISO_ROCK['PR']):
        super().__init__()
        self.symmetry = 'ISO'
        self.set_compliance(E, PR)
        self.get_stiffness()
        self.E = E
        self.PR = PR

    def set_compliance(self, E, PR):
        """Set compliance tensor from given elastic moduli: Young Modulus(E) as Poisson Ratio(PR)"""
        c = self.compliance.tensor
        c[0, 0], c[0, 1], c[0, 2] = 1 / E, -PR / E, -PR / E
        c[1, 0], c[1, 1], c[1, 2] = -PR / E, 1 / E, -PR / E
        c[2, 0], c[2, 1], c[2, 2] = -PR / E, -PR / E, 1 / E
        c[3, 3] = 2 * (1 + PR) / E
        c[4, 4] = 2 * (1 + PR) / E
        c[5, 5] = 2 * (1 + PR) / E


class TIVRock(Rock):
    """Class for anisotropic rocks with TIV symmetry"""
    def __init__(self,
                 Ev=TIV_ROCK['Ev'],
                 Eh=TIV_ROCK['Eh'],
                 PRv=TIV_ROCK['PRv'],
                 PRhh=TIV_ROCK['PRhh'],
                 Gv=TIV_ROCK['Gv']):
        super().__init__()
        self.symmetry = 'TIV'
        self.set_compliance(Ev, Eh, PRv, PRhh, Gv)
        self.get_stiffness()
        C = self.stiffness.tensor
        self.PRhv = - (C[0, 2]*(C[0, 1] - C[0, 0])) / (C[0, 0] * C[2, 2] - C[0, 2]**2)
        self.Gv = Gv
        self.GvHuber = np.sqrt(Ev * Eh) / (2 * (1 + np.sqrt(PRv * self.PRhv)))

    def set_compliance(self, Ev, Eh, PRv, PRhh, Gv):
        c = self.compliance.tensor
        c[0, 0], c[0, 1], c[0, 2] = 1 / Eh, -PRhh / Eh, -PRv / Ev
        c[1, 0], c[1, 1], c[1, 2] = -PRhh / Eh, 1 / Eh, -PRv / Ev
        c[2, 0], c[2, 1], c[2, 2] = -PRv / Ev, -PRv / Ev, 1 / Ev
        c[3, 3] = 1 / Gv
        c[4, 4] = 1 / Gv
        c[5, 5] = 2 * (1 + PRhh) / Eh


class ORTRock(Rock):
    def __init__(self,
                 Ex=ORT_ROCK['Ex'],
                 Ey=ORT_ROCK['Ey'],
                 Ez=ORT_ROCK['Ez'],
                 PRyx=ORT_ROCK['PRyx'],
                 PRzx=ORT_ROCK['PRzx'],
                 PRzy=ORT_ROCK['PRzy'],
                 Gyz=ORT_ROCK['Gyz'],
                 Gxz=ORT_ROCK['Gxz'],
                 Gxy=ORT_ROCK['Gxy']):
        super().__init__()
        self.symmetry = 'ORT'
        self.set_compliance(Ex, Ey, Ez, PRyx, PRzx, PRzy, Gyz, Gxz, Gxy)
        self.get_stiffness()
        #  :todo: add Huber approx check

    def set_compliance(self, Ex, Ey, Ez, PRyx, PRzx, PRzy, Gyz, Gxz, Gxy):
        c = self.compliance.tensor
        c[0, 0], c[0, 1], c[0, 2] = 1 / Ex, -PRyx / Ey, -PRzx / Ez
        c[1, 0], c[1, 1], c[1, 2] = -PRyx / Ex, 1 / Ey, -PRzy / Ez
        c[2, 0], c[2, 1], c[2, 2] = -PRzx / Ex, -PRzy / Ey, 1 / Ez
        c[3, 3] = 1 / Gyz
        c[4, 4] = 1 / Gxz
        c[5, 5] = 1 / Gxy


