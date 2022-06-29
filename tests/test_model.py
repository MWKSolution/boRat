import unittest
import numpy as np
from boRat import model, stress, rock, wellbore


class TestStressRotation(unittest.TestCase):

    def setUp(self) -> None:
        self.s = stress.Stress()
        self.s.set_from_PCS(SH=1, Sh=2, Sz=3)
        self.r = rock.ISORock()

    def test_SH_E_horizontal_E(self):
        self.dip = rock.FormationDip(dip=0, dir=0)
        self.wbo = wellbore.WellboreOrientation(hazi=90, hdev=90)
        self.m = model.BoreholeModel(self.s, self.r, self.dip, self.wbo, 90, 1)
        result = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
        self.assertTrue(np.isclose(self.m.stress_toh.stress, result).all())

    def test_SH_E_horizontal_S(self):
        self.dip = rock.FormationDip(dip=0, dir=0)
        self.wbo = wellbore.WellboreOrientation(hazi=180, hdev=90)
        self.m = model.BoreholeModel(self.s, self.r, self.dip, self.wbo, 90, 1)
        result = np.array([[3, 0, 0], [0, 1, 0], [0, 0, 2]])
        self.assertTrue(np.isclose(self.m.stress_toh.stress, result).all())

    def test_SH_NE_horizontal_NE(self):
        self.dip = rock.FormationDip(dip=0, dir=0)
        self.wbo = wellbore.WellboreOrientation(hazi=45, hdev=90)
        self.m = model.BoreholeModel(self.s, self.r, self.dip, self.wbo, 45, 1)
        result = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
        self.assertTrue(np.isclose(self.m.stress_toh.stress, result).all())

    def test_SH_NE_horizontal_SW(self):
        self.dip = rock.FormationDip(dip=0, dir=0)
        self.wbo = wellbore.WellboreOrientation(hazi=225, hdev=90)
        self.m = model.BoreholeModel(self.s, self.r, self.dip, self.wbo, 45, 1)
        result = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
        self.assertTrue(np.isclose(self.m.stress_toh.stress, result).all())


class TestRockRotation(unittest.TestCase):

    def setUp(self) -> None:
        self.s = stress.Stress()
        self.s.set_from_PCS(SH=1, Sh=2, Sz=3)
        self.r = rock.TIVRock()

    def test_well_normal_to_bedding_0(self):
        self.dip = rock.FormationDip(dip=0, dir=0)
        self.wbo = wellbore.WellboreOrientation(hazi=0, hdev=0)
        self.m = model.BoreholeModel(self.s, self.r, self.dip, self.wbo, 0, 1)
        self.assertTrue(np.isclose(self.m.rock.compliance.tensor, self.m.rock_toh.compliance.tensor).all())

    def test_well_normal_to_bedding_45(self):
        self.dip = rock.FormationDip(dip=45, dir=180)
        self.wbo = wellbore.WellboreOrientation(hazi=0, hdev=45)
        self.m = model.BoreholeModel(self.s, self.r, self.dip, self.wbo, 0, 1)
        self.assertTrue(np.isclose(self.m.rock.compliance.tensor, self.m.rock_toh.compliance.tensor).all())

    def test_well_normal_to_bedding_90(self):
        self.dip = rock.FormationDip(dip=90, dir=180)
        self.wbo = wellbore.WellboreOrientation(hazi=0, hdev=90)
        self.m = model.BoreholeModel(self.s, self.r, self.dip, self.wbo, 0, 1)
        self.assertTrue(np.isclose(self.m.rock.compliance.tensor, self.m.rock_toh.compliance.tensor).all())

    def test_well_normal_to_bedding_45_45(self):
        self.dip = rock.FormationDip(dip=45, dir=225)
        self.wbo = wellbore.WellboreOrientation(hazi=45, hdev=45)
        self.m = model.BoreholeModel(self.s, self.r, self.dip, self.wbo, 0, 1)
        self.assertTrue(np.isclose(self.m.rock.compliance.tensor, self.m.rock_toh.compliance.tensor).all())

    def test_well_normal_to_bedding_50_50(self):
        self.dip = rock.FormationDip(dip=50, dir=230)
        self.wbo = wellbore.WellboreOrientation(hazi=50, hdev=50)
        self.m = model.BoreholeModel(self.s, self.r, self.dip, self.wbo, 0, 1)
        self.assertTrue(np.isclose(self.m.rock.compliance.tensor, self.m.rock_toh.compliance.tensor).all())


class TestBeltramiMichell(unittest.TestCase):
    def setUp(self) -> None:
        self.stress_pcs = stress.Stress()
        self.stress_pcs.set_from_PCS(SH=20, Sh=10, Sz=30)
        self.iso_rock = rock.ISORock()

    def test_model(self):

        for p in range(0, 91, 10):
            for r in range(0, 181, 10):
                for h in range(0, 181, 10):
                    for d in range(0, 91, 10):
                        dip = rock.FormationDip(dip=p, dir=r)
                        wbo = wellbore.WellboreOrientation(hazi=h, hdev=d)
                        self.modelBM = model.BoreholeModel(self.stress_pcs, self.iso_rock, dip, wbo, 0, 5, hoop_model='beltrami-michell')
                        self.modelK = model.BoreholeModel(self.stress_pcs, self.iso_rock, dip, wbo, 0, 5, hoop_model='kirsch')

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