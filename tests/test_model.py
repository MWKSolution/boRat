import unittest
import numpy as np
from boRat import model, stress, rock, wellbore


class TestBeltramiMichell(unittest.TestCase):
    def setUp(self) -> None:
        self.stress = stress.Stress.from_PCS(SH=20, Sh=10, Sv=30, SHazi=15)

    def test_model(self):

        for p in range(0, 91, 30):
            for r in range(0, 181, 30):
                dip = rock.FormationDip(dip=p, dir=r)
                _rock = rock.Rock.ISO_from_moduli(dip=dip)
                for h in range(0, 181, 30):
                    for d in range(0, 91, 30):
                        wbo = wellbore.Wellbore(hazi=h, hdev=d, Pw=5)
                        self.modelBM = model.BoreholeModel(self.stress, _rock, wbo, hoop_model='beltrami-michell')
                        self.modelK = model.BoreholeModel(self.stress, _rock, wbo, hoop_model='kirsch')
                        print(f'Test for: dip {p}, dir {r}, azi {h}, dev {d} ', end='')
                        for theta in range(0, 180, 20):
                            hoopBM = self.modelBM.get_hoop_stress(theta)
                            hoopK = self.modelK.get_hoop_stress(theta)
                            self.assertTrue(np.isclose(hoopBM.stress, hoopK.stress, atol=1e-3).all())
                            print('.', end='')
                        print('OK')


if __name__ == '__main__':
    unittest.main(verbosity=3)