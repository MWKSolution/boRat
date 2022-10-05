import numpy as np
from numpy.polynomial.polynomial import polyroots
from boRat.tensor import TensorVoigt
from boRat.stress import Stress
from boRat.config import __log__


class HoopStress:
    def __init__(self, rock, far_field_stress, Pw):
        self.far_field_stress = far_field_stress
        self.Pw = Pw

    def get_borehole_hoop_stress(self, theta):
        hoop_stress = Stress()
        return hoop_stress


class BeltramiMichell(HoopStress):
    def __init__(self, rock, far_field_stress, Pw):
        # __log__.info('-------------------- Beltrami-Michell --------------------')
        super().__init__(rock, far_field_stress, Pw)
        self.compliance = rock.compliance
        self.beta = self.get_reduced_strain_coeff(self.compliance)
        self.beta.clean()
        self.a = self.get_polynomial_coeffs(self.beta.tensor)
        self.all_roots = polyroots(self.a)
        self.roots = self.get_conjugate_roots()
        # :todo: sprawdzic to!
        self.mi1,  self.mi2, self.mi3 = self.roots[0], self.roots[1], self.roots[2]
        # self.mi1 = 0 + 1j
        # self.mi2 = 0 + 1j
        # self.mi3 = 0 + 1j
        self.la1 = - (self.I3(self.mi1) / self.I2(self.mi1))
        self.la2 = - (self.I3(self.mi2) / self.I2(self.mi2))
        self.la3 = - (self.I3(self.mi3) / self.I4(self.mi3))

        # __log__.debug(f'BM: compliance: \n{self.compliance!s}')
        # __log__.debug(f'BM: beta: \n{self.beta!s}')
        # __log__.debug('BM: a: ' + ' '.join([f'\n{a}' for a in self.a]))
        __log__.info('BM: all roots:' + ''.join([f'\n{r}' for r in self.all_roots]))
        __log__.info('BM: conjugate roots:' + ''.join([f'\n{r}' for r in self.roots]))
        __log__.info(f'BM: mi1, mi2, mi3: \n{self.mi1}, \n{self.mi2}, \n{self.mi3}')
        # __log__.debug(f'BM: la1, la2, la3: \n{self.la1}, \n{self.la2}, \n{self.la3}')
        # __log__.info('-------------------- Beltrami-Michell --------------------')

    def get_conjugate_roots(self):
        """Get only 3 conjugates"""
        con_roots = []
        for i in self.all_roots:
            if i.imag >= 0:
                con_roots.append(i)
        return con_roots

    @staticmethod
    def get_reduced_strain_coeff(t):
        """Reduced elastic constants"""
        beta = TensorVoigt()
        for i in range(6):
            for j in range(6):
                a = t.tensor
                beta.tensor[i, j] = a[i, j] - ((a[i, 2] * a[j, 2]) / a[2, 2])
        B11, B55 = beta.tensor[0, 0], beta.tensor[4, 4]
        # __log__.info(f'B11 = {B11:.4f}, B55 = {B55:.4f}')
        # if np.isclose(B11, B55):
        #     __log__.warning('B11 == B55 !!!')

        return beta

    @staticmethod
    def get_polynomial_coeffs(b):
        a = np.zeros(7, dtype=np.complex64)
        a[6] = b[0, 0] * b[4, 4] - b[0, 4] * b[0, 4] + 0j
        a[5] = 2 * b[0, 4] * (b[0, 3] + b[4, 5]) - 2 * (b[0, 5] * b[4, 4] + b[0, 0] * b[3, 4]) + 0j
        a[4] = b[4, 4] * (2 * b[0, 1] + b[5, 5]) + 4 * b[0, 5] * b[3, 4] + b[0, 0] * b[3, 3] - \
               (b[0, 3] + b[4, 5]) ** 2 - 2 * b[0, 4] * (b[1, 4] + b[3, 5]) + 0j
        a[3] = -2 * b[1, 5] * b[4, 4] - 2 * b[3, 4] * (2 * b[0, 1] + b[5, 5]) - 2 * b[0, 5] * b[3, 3] + \
               2 * b[0, 4] * b[1, 3] + 2 * (b[0, 3] + b[4, 5]) * (b[1, 4] + b[3, 5]) + 0j
        a[2] = b[1, 1] * b[4, 4] + 4 * b[1, 5] * b[3, 4] + b[3, 3] * (2 * b[0, 1] + b[5, 5]) - \
               2 * b[1, 3] * (b[0, 3] + b[4, 5]) - (b[1, 4] + b[3, 5]) ** 2 + 0j
        a[1] = -2 * b[1, 1] * b[3, 4] - 2 * b[1, 5] * b[3, 3] + 2 * b[1, 3] * (b[1, 4] + b[3, 5]) + 0j
        a[0] = b[1, 1] * b[3, 3] - b[1, 3] * b[1, 3] + 0j
        return a

    def I4(self, x):
        b = self.beta.tensor
        return b[0, 0] * (x ** 4) - 2 * b[0, 5] * (x ** 3) + (2 * b[0, 1] + b[5, 5]) * (x ** 2) - 2 * b[1, 5] * x + b[1, 1]

    def I3(self, x):
        b = self.beta.tensor
        return b[0, 4] * (x ** 3) - (b[0, 3] + b[4, 5]) * (x ** 2) + (b[1, 4] + b[3, 5]) * x - b[1, 3]

    def I2(self, x):
        b = self.beta.tensor
        return b[4, 4] * (x ** 2) - 2 * b[3, 4] * x + b[3, 3]

    # @timer
    def get_borehole_hoop_stress(self, theta):
        t = np.radians(theta)
        s = self.far_field_stress.stress
        c = self.compliance.tensor
        p = self.Pw
        sint, cost = np.sin(t), np.cos(t)
        mi1, mi2, mi3 = self.mi1, self.mi2, self.mi3
        la1, la2, la3 = self.la1, self.la2, self.la3

        D = ((p - s[0, 0]) * cost - s[0, 1] * sint) + (-(p - s[0, 0]) * sint - s[0, 1] * cost) * 1j
        E = (-(p - s[1, 1]) * sint + s[0, 1] * cost) + (-(p - s[1, 1]) * cost - s[0, 1] * cost) * 1j
        F = (-s[2, 0] * cost - s[2, 1] * sint) + (s[2, 0] * sint - s[2, 1] * cost) * 1j
        _x = mi2 - mi1 + la2 * la3 * (mi1 - mi3) + la1 * la3 * (mi3 - mi2)
        G1 = (mi1 * cost - sint) * _x
        G2 = (mi2 * cost - sint) * _x
        G3 = (mi3 * cost - sint) * _x

        fi1 = ((D * ((la2 * la3) - 1)) + (E * (mi2 - (la2 * la3 * mi3))) + (F * la3 * (mi3 - mi2))) / (2 * G1)
        fi2 = ((D * (1 - (la1 * la3))) + (E * ((la1 * la3 * mi3) - mi1)) + (F * la3 * (mi1 - mi3))) / (2 * G2)
        fi3 = ((D * (la1 - la2)) + (E * ((mi1 * la2) - (mi2 * la1))) + (F * (mi2 - mi1))) / (2 * G3)

        dsig_xx = 2 * (((mi1 ** 2) * fi1) + ((mi2 ** 2) * fi2) + (la3 * (mi3 ** 2) * fi3)).real
        dsig_yy = 2 * (fi1 + fi2 + (la3 * fi3)).real
        dtau_xy = -2 * ((mi1 * fi1) + (mi2 * fi2) + (la3 * mi3 * fi3)).real
        dtau_xz = 2 * ((la1 * mi1 * fi1) + (la2 * mi2 * fi2) + (mi3 * fi3)).real
        dtau_yz = -2 * ((la1 * fi1) + (la2 * fi2) + fi3).real
        dsig_zz = -((c[2, 0] * dsig_xx + c[2, 1] * dsig_yy + c[2, 3] * dtau_yz + c[2, 4] * dtau_xz + c[
            2, 5] * dtau_xy) / c[2, 2])

        hoop_stress = Stress()
        dh = Stress()
        dh.stress = np.array([[dsig_xx, dtau_xy, dtau_xz],
                              [dtau_xy, dsig_yy, dtau_yz],
                              [dtau_xz, dtau_yz, dsig_zz]])

        hoop_stress.stress = np.add(s, dh.stress)
        hoop_stress_cyl = hoop_stress.cart2cyl(theta)
        del hoop_stress

        return hoop_stress_cyl


class Kirsch(HoopStress):
    def __init__(self, rock, far_field_stress, Pw):
        super().__init__(rock, far_field_stress, Pw)
        c = rock.compliance.tensor
        self.PR = -c[0, 1] / c[0, 0]  # :todo: averaging of PR fot non ISO ???

    def get_borehole_hoop_stress(self, theta):
        t = np.radians(theta)
        s = self.far_field_stress.stress
        sig_rr = self.Pw
        sig_tt = s[0, 0] + s[1, 1] - 2 * (s[0, 0] - s[1, 1]) * np.cos(2 * t) - 4 * s[0, 1] * np.sin(2 * t) - self.Pw
        # sig_tt = s[0, 0] + s[1, 1] - 2 * (s[0, 0] - s[1, 1]) * np.cos(2 * t) - 4 * s[0, 1] * np.sin(2 * t)
        sig_zz = s[2, 2] - self.PR * (2 * ((s[0, 0] - s[1, 1]) * np.cos(2 * t)) + 4 * s[0, 1] * np.sin(2 * t))
        tau_rt = 0
        tau_rz = 0
        tau_tz = 2 * (s[1, 2] * np.cos(t) - s[0, 2] * np.sin(t))

        hoop_stress = Stress()
        hoop_stress.stress = np.array([[sig_rr, tau_rt, tau_rz],
                                       [tau_rt, sig_tt, tau_tz],
                                       [tau_rz, tau_tz, sig_zz]])
        return hoop_stress
