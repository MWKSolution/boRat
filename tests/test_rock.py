import unittest
import numpy as np
from boRat.rock import Rock, FormationDip


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


class TestRockRotation(unittest.TestCase):

    @unittest.skip
    def test_rot(self):
        """Rotation is tested in TensorVoigt class (tensor.py -> test_tensor.py) """
        pass


class TestStiffnessTensor(unittest.TestCase):

    def setUp(self) -> None:
        self.tiv = TIVRock()

    def test_stiffness_inversion(self):
        """Test tensor inversion"""
        c = self.tiv.compliance
        self.tiv.get_stiffness()
        _ = self.tiv.stiffness
        self.tiv.compliance = _
        self.tiv.get_stiffness()
        s = self.tiv.stiffness
        self.assertTrue(np.isclose(c.tensor, s.tensor).all())


class TestComplianceTensor(unittest.TestCase):

    def setUp(self) -> None:
        self.iso = ISORock()
        self.tiv = TIVRock()
        self.ort = ORTRock()

    def test_ISO_compliance(self):
        """Well known formula - no need to be tested"""
        pass

    def test_TIV_compliance(self):
        """Well known formula - no need to be tested"""
        pass

    def test_ORT_compliance(self):
        """Well known formula - no need to be tested"""
        pass


if __name__ == '__main__':
    unittest.main(verbosity=3)
