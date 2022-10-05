import unittest
import numpy as np
from boRat import model, stress, rock, wellbore


class TestBeltramiMichell(unittest.TestCase):
    def setUp(self) -> None:
        self.stress = stress.Stress.from_PCS(SH=20, Sh=10, Sv=30, SHazi=0)


    def test_model(self):

        for p in range(0, 91, 10):
            for r in range(0, 181, 10):
                for h in range(0, 181, 10):
                    for d in range(0, 91, 10):
                        dip = rock.FormationDip(dip=p, dir=r)
                        wbo = wellbore.Wellbore(hazi=h, hdev=d, Pw=5)
                        _rock = rock.Rock.ISO_from_moduli(dip=dip)
                        self.modelBM = model.BoreholeModel(self.stress, _rock, wbo, hoop_model='beltrami-michell')
                        self.modelK = model.BoreholeModel(self.stress, _rock, wbo, hoop_model='kirsch')

                        print(f'dip {p}, dir {r}, azi {h}, dev {d} ', end='')
                        for theta in range(0, 360, 10):
                            hoopBM = self.modelBM.get_hoop_stress(theta)
                            hoopBM.clean()
                            hoopK = self.modelK.get_hoop_stress(theta)
                            self.assertTrue(np.isclose(hoopBM.stress, hoopK.stress, atol=1e-2).all())
                            print('.', end='')
                        print('')


if __name__ == '__main__':
    unittest.main(verbosity=3)