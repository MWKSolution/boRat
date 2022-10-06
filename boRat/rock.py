import numpy as np
from boRat.tensor import Compliance, Stiffness
from scipy.spatial.transform import Rotation as Rot
from boRat.config import __log__

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
        self.vector = self.get_normal_vector_NEV()
        __log__.debug(f'Formation dip-> dip:{dip} dir:{dir} degs')

    def get_normal_vector_NEV(self):
        """Vector perpendicular to bedding. Since for flat bedding it is pointing down (Z axis direction) it has to be rotated with -dip value!"""
        init = np.array([0, 0, 1])
        rotation = Rot.from_euler('ZYX', [self.dir, -self.dip, 0], degrees=True)
        return rotation.apply(init)

    def __repr__(self):
        return f'FormationDip({self.dip!s}, {self.dir!s})'

    def __str__(self):
        return f'FormationDip(Dip angle: {self.dip!s}, Dip direction: {self.dir!s}, Normal vector: {self.vector!s} NEV)'


class Rock:
    """Class describing rock mechanical properties:
        compliance - compliance tensor in Voigt notation,
        stiffness - stiffness tensor in Voigt notation,
        ..."""

    # :todo: lampiere verification and other verifications for E, PR and G
    def __init__(self, compliance=Compliance(), dip=FormationDip(), symmetry=None):
        self.compliance = compliance
        self.rot_PSC_to_NEV(dip.dip, dip.dir)
        self.stiffness = Stiffness(self.get_stiffness())
        self.dip = dip
        self.symmetry = symmetry

    def get_stiffness(self):
        """Get stiffness from compliance. By definition, it is inversion of compliance."""
        return np.linalg.inv(self.compliance.tensor)

    def rot_PSC_to_NEV(self, dip, dir):
        rotation = Rot.from_euler('ZY', [dir, -dip], degrees=True)
        self.rot_inplace(rotation)

    def rot_NEV_to_TOH(self, hazi, hdev):
        rotation = Rot.from_euler('YZ', [-hdev, -hazi], degrees=True)
        return self.rot(rotation)

    def rot(self, rotation):
        """Rotate compliance tensor with given angles and get stiffness tensor."""
        rotated = Rock(self.compliance.rot(rotation))
        rotated.symmetry = self.symmetry
        return rotated

    def rot_inplace(self, rotation):
        self.compliance.rot_inplace(rotation)

    def clean(self):
        self.compliance.clean()
        self.stiffness.clean()

    def __repr__(self):
        return f'Rock:(\nCompliance:({self.compliance!s})\nStiffness:({self.stiffness!s}))'

    @classmethod
    def ISO_from_moduli(cls,
                        E=ISO_ROCK['E'],
                        PR=ISO_ROCK['PR'],
                        dip=FormationDip()):
        """Set compliance tensor from given elastic moduli: Young Modulus(E) as Poisson Ratio(PR)"""
        # self.set_compliance(E, PR)
        # self.get_stiffness()
        # self.E = E
        # self.PR = PR
        c = np.zeros((6, 6), dtype=np.float64)
        c[0, 0], c[0, 1], c[0, 2] = 1 / E, -PR / E, -PR / E
        c[1, 0], c[1, 1], c[1, 2] = -PR / E, 1 / E, -PR / E
        c[2, 0], c[2, 1], c[2, 2] = -PR / E, -PR / E, 1 / E
        c[3, 3], c[4, 4], c[5, 5] = 2 * (1 + PR) / E, 2 * (1 + PR) / E, 2 * (1 + PR) / E
        comp = Compliance(tensor=c)
        return cls(comp, dip, 'ISO')

    @classmethod
    def TIV_from_moduli(cls,
                        Ev=TIV_ROCK['Ev'],
                        Eh=TIV_ROCK['Eh'],
                        PRv=TIV_ROCK['PRv'],
                        PRhh=TIV_ROCK['PRhh'],
                        Gv=TIV_ROCK['Gv'],
                        dip=FormationDip()):
        # C = self.stiffness.tensor
        # self.PRhv = - (C[0, 2]*(C[0, 1] - C[0, 0])) / (C[0, 0] * C[2, 2] - C[0, 2]**2)
        # self.Gv = Gv
        # self.GvHuber = np.sqrt(Ev * Eh) / (2 * (1 + np.sqrt(PRv * self.PRhv)))
        c = np.zeros((6, 6), dtype=np.float64)
        c[0, 0], c[0, 1], c[0, 2] = 1 / Eh, -PRhh / Eh, -PRv / Ev
        c[1, 0], c[1, 1], c[1, 2] = -PRhh / Eh, 1 / Eh, -PRv / Ev
        c[2, 0], c[2, 1], c[2, 2] = -PRv / Ev, -PRv / Ev, 1 / Ev
        c[3, 3], c[4, 4], c[5, 5] = 1 / Gv, 1 / Gv, 2 * (1 + PRhh) / Eh
        comp = Compliance(tensor=c)
        return cls(comp, dip, 'TIV')

    @classmethod
    def ORT_from_moduli(cls,
                        Ex=ORT_ROCK['Ex'],
                        Ey=ORT_ROCK['Ey'],
                        Ez=ORT_ROCK['Ez'],
                        PRyx=ORT_ROCK['PRyx'],
                        PRzx=ORT_ROCK['PRzx'],
                        PRzy=ORT_ROCK['PRzy'],
                        Gyz=ORT_ROCK['Gyz'],
                        Gxz=ORT_ROCK['Gxz'],
                        Gxy=ORT_ROCK['Gxy'],
                        dip=FormationDip()):
        #  :todo: add Huber approx check
        c = np.zeros((6, 6), dtype=np.float64)
        c[0, 0], c[0, 1], c[0, 2] = 1 / Ex, -PRyx / Ey, -PRzx / Ez
        c[1, 0], c[1, 1], c[1, 2] = -PRyx / Ex, 1 / Ey, -PRzy / Ez
        c[2, 0], c[2, 1], c[2, 2] = -PRzx / Ex, -PRzy / Ey, 1 / Ez
        c[3, 3], c[4, 4], c[5, 5] = 1 / Gyz, 1 / Gxz, 1 / Gxy
        comp = Compliance(tensor=c)
        return cls(comp, dip, 'ORT')


if __name__ == '__main__':
    dip = FormationDip(dip=5, dir=5)
    print(f'str : {dip!s}')
    print(f'repr: {dip!r}')

    iso = Rock.ISO_from_moduli(**ISO_ROCK, dip=dip)
    print(iso)
    tiv = Rock.TIV_from_moduli(**TIV_ROCK, dip=dip)
    print(tiv)
    ort = Rock.ORT_from_moduli(**ORT_ROCK, dip=dip)
    print(ort)

