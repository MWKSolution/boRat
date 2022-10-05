import unittest
import numpy as np
from boRat.rock import Rock, FormationDip
from boRat.tensor import Compliance


class TestNormalVector(unittest.TestCase):

    def setUp(self) -> None:
        self.horizontal = np.array([0, 0, 1])
        self.vertical_N = np.array([-1, 0, 0])
        self.vertical_E = np.array([0, -1, 0])
        self.vertical_S = np.array([1, 0, 0])
        self.vertical_W = np.array([0, 1, 0])

    def test_horizontal(self):
        """Flat bedding"""
        dip = FormationDip().vector
        self.assertTrue(np.isclose(dip, self.horizontal).all())

    def test_dipping_north(self):
        """dip=90, dipping North"""
        dip = FormationDip(dip=90, dir=0).vector
        self.assertTrue(np.isclose(dip, self.vertical_N).all())

    def test_dipping_east(self):
        """dip=90, dipping East"""
        dip = FormationDip(dip=90, dir=90).vector
        self.assertTrue(np.isclose(dip, self.vertical_E).all())

    def test_dipping_south(self):
        """dip=90, dipping South"""
        dip = FormationDip(dip=90, dir=180).vector
        self.assertTrue(np.isclose(dip, self.vertical_S).all())

    def test_dipping_west(self):
        """dip=90, dipping West"""
        dip = FormationDip(dip=90, dir=270).vector
        self.assertTrue(np.isclose(dip, self.vertical_W).all())


class TestRockCombinedRotation(unittest.TestCase):

    def setUp(self) -> None:
        self.iso = Rock.ISO_from_moduli()
        self.tiv = Rock.TIV_from_moduli()
        self.ort = Rock.ORT_from_moduli()

    def test_rot_1(self):
        """Formation dip and wellbore orientation aligned - should get initial tensor"""
        ortc = Compliance(self.ort.compliance.tensor)

        self.ort.rot_PSC_to_NEV(dip=25, dir=35)  # in place - NEV coords
        rotated = self.ort.rot_NEV_to_TOH(hazi=35+180, hdev=25)  # TOH coords - wellbore perpendicular to bedding
        self.assertTrue(np.isclose(rotated.compliance.tensor, ortc.tensor).all())

    def test_rot_2(self):
        """Formation dip and wellbore orientation aligned - should get initial tensor"""
        ortc = Compliance(self.ort.compliance.tensor)

        self.ort.rot_PSC_to_NEV(dip=90, dir=90)  # in place - NEV coords
        rotated = self.ort.rot_NEV_to_TOH(hazi=90+180, hdev=90)  # TOH coords - wellbore perpendicular to bedding

        self.assertTrue(np.isclose(rotated.compliance.tensor, ortc.tensor).all())

    def test_rot_3(self):
        """Formation dip and wellbore orientation aligned - should get initial tensor"""
        ortc = Compliance(self.ort.compliance.tensor)

        self.ort.rot_PSC_to_NEV(dip=0, dir=45)  # in place - NEV coords
        rotated = self.ort.rot_NEV_to_TOH(hazi=45+180, hdev=0)  # TOH coords - wellbore perpendicular to bedding

        self.assertTrue(np.isclose(rotated.compliance.tensor, ortc.tensor).all())


class TestStiffnessTensor(unittest.TestCase):

    def test_stiffness(self):
        """Test tensor inversion"""
        print('Stiffness is tested when tesnor rotations are tested')


class TestComplianceTensor(unittest.TestCase):

    def test_stiffness(self):
        """Test tensor inversion"""
        print('Compliance is tested when tesnor rotations are tested')


if __name__ == '__main__':
    unittest.main(verbosity=3)
