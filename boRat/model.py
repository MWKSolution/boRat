import numpy as np

from boRat import Stress
from boRat import Rock, FormationDip
from boRat.beltrami_michell import BeltramiMichell, Kirsch
from boRat.config import __log__
from boRat.plots import model_plot, stress_plot, bedding_plot, all_plot, compare_stresses_plot
from boRat import Wellbore


class BoreholeModel:

    class ModelInitError(Exception):
        pass

    def __init__(self,
                 stress: Stress,
                 rock: Rock,
                 wbo: Wellbore,
                 hoop_model='beltrami-michell',  # or 'kirsch'
                 clean=True):
        self.stressNEV = stress
        self.rockNEV = rock
        self.wbo = wbo
        self.angle = self.get_angle(self.wbo.vector, self.rockNEV.dip.vector)
        if clean:
            self.stressNEV.clean()
            self.rockNEV.clean()
        self.stressTOH = self.stressNEV.rot_NEV_to_TOH(self.wbo.orien.hazi, self.wbo.orien.hdev)
        self.rockTOH = self.rockNEV.rot_NEV_to_TOH(self.wbo.orien.hazi, self.wbo.orien.hdev)
        if clean:
            self.stressTOH.clean()
            self.rockTOH.clean()

        # __log__.info('-------------------- model description start --------------------')
        # __log__.info(f'Principal stresses [Mpa]: {self.stress_pcs!s}')
        # __log__.info(f'SH azimuth [deg]: {self.SHAzi:.2f}')
        # __log__.info(f'Mud Pressure [Mpa]: {self.Pw:.2f}')
        # __log__.info(f'{self.rock!s}')
        # if isinstance(self.rock, TIVRock):
        #     __log__.info(f"TIV: PRhv= {self.rock.PRhv:.4f}; Gv Huber's approx.={self.rock.GvHuber:.4f}, Gv= {self.rock.Gv:.4f}")
        # __log__.info(f'{self.dip!s}')
        # __log__.info(f'{self.wbo!s}')
        # __log__.info(f'Angle between bedding plane and wellbore axis: {self.angle:.4f}')
        # __log__.debug(f'Stresses in NEV coordinates: {self.stress_nev!s}')
        # __log__.debug(f'Stresses in TOH coordinates: {self.stress_toh!s}')
        # __log__.debug(f'Rock tensors in NEV coordinates: {self.rock_nev!s}')
        # __log__.debug(f'Rock tensors in TOH coordinates: {self.rock_toh!s}')
        # __log__.info(f'Anisotropy level: {self.rock.symmetry!s}, Model: {hoop_model!s}')
        # __log__.info('-------------------- model description end --------------------')

        if hoop_model == 'beltrami-michell':
            self.Hoop = BeltramiMichell
        elif hoop_model == 'kirsch' and self.rockNEV.symmetry == 'ISO':
            self.Hoop = Kirsch
        else:
            self.Hoop = None
            pass
            # :todo: raise sth later...

        self.model = self.Hoop(self.rockTOH, self.stressTOH, self.wbo.Pw, clean=clean)

    @staticmethod
    def get_angle(v1, v2):
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        dot = np.dot(v1, v2)
        angle = np.arccos(dot / norm)
        return np.degrees(angle)

    def get_hoop_stress(self, theta):
        hoop_stress = self.model.get_borehole_hoop_stress(theta)
        return hoop_stress

    def show_model(self):
        mp = model_plot(self)
        mp.show()

    def show_bedding(self):
        bp = bedding_plot(self)
        bp.show()

    def show_stress(self):
        sp = stress_plot(self)
        sp.show()

    def show_all(self):
        allp = all_plot(self)
        allp.show()

    def compare_stresses_with(self, model2):
        comp = compare_stresses_plot(self, model2)
        comp.show()


if __name__ == '__main__':

    stress = Stress.from_PCS(SH=20, Sh=10, Sv=30, SHazi=10)

    ISO_ROCK = dict(E=30.14, PR=0.079)
    TIV_ROCK = dict(Ev=15.42, Eh=31.17, PRv=0.32, PRhh=0.079, Gv=7.05)
    ORT_ROCK = dict(Ex=30, Ey=20, Ez=15, PRyx=0.3, PRzx=0.2, PRzy=0.1, Gyz=6, Gxz=7, Gxy=8)

    dip = FormationDip(dip=30, dir=30)

    # # rock = Rock.ISO_from_moduli(**ISO_ROCK, dip=dip)
    rock = Rock.TIV_from_moduli(**TIV_ROCK, dip=dip)
    # # rock = Rock.ORT_from_moduli(**ORT_ROCK, dip=dip)
    #
    wbo = Wellbore(hazi=45, hdev=45, Pw=5)
    #
    # # model = BoreholeModel(stress, rock, wbo, hoop_model='kirsch')
    # # model.show_all()
    #
    modelBM = BoreholeModel(stress, rock, wbo, hoop_model='beltrami-michell', clean=False)
    # # model.compare_stresses_with(modelBM)
    modelBM.show_stress()





