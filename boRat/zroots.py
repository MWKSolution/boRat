"""Module based on:
Numerical Recipes in C: The Art of Scientific Computing
William H. Press, Saul A. Teukolsky, William T. Vetterling, Brian P. Flannery, Michael Metcalf
1992 Cambridge University Press.
9.5 Roots of Polynomials, p 369 - 379"""

import numpy as np
from math import fabs


class LaguerreIterationError(Exception):
    pass


def laguerre(a, m, x):
    """The following routine implements the Laguerre method to find one root of a given polynomial of degree m, whose coefficients a can be complex.
    a[] - m+1 coefficients of complex polynomial
    m   - degree of polynomial
    x   - initial guess"""

    EPSS = 1e-7  # Estimated fractional round-off error.
    FRAC = (0.5, 0.25, 0.75, 0.13, 0.38, 0.62, 0.88, 1.0)  # Fractions used to break a limit cycle.
    MR, MT = 8, 10
    MAXIT = MT * MR
    # We try to break (rare) limit cycles with MR different fractional values, once every MT steps, for MAXIT total allowed iterations.

    # Given the degree m and the m+1 complex coefficients a[0, ..., m] of the polynomial SUM(i=0 to m)a(i)x**i
    # and given a complex value x, this routine improves x by Laguerre’s method until it converges,
    # within the achievable round-off limit, to a root of the given polynomial.
    # The number of iterations taken is returned.

    for iter in range(1, MAXIT+1):  # Loop over iterations up to allowed maximum.
        b = a[m]
        err = abs(b)
        abx = abs(x)
        d, f = 0 + 0j, 0 + 0j
        for j in range(m-1, -1, -1):  # Efficient computation of the polynomial and its first two derivatives.
            f = x * f + d
            d = x * d + b
            b = x * b + a[j]
            err = abs(b) + abx * err
        err = EPSS * err  # Estimate of round-off error in evaluating polynomial.
        if abs(b) <= err:
            return x, iter              # We are on the root.
        else:  # The generic case: use Laguerre’s formula.
            g = d / b
            g2 = g * g
            h = g2 - 2 * f / b
            sq = (float(m - 1) * (float(m) * h - g2))**(1/2)
            gp = g + sq
            gm = g - sq
            abp = abs(gp)
            abm = abs(gm)
            if abp < abm:
                gp = gm
            if max(abp, abm) > 0:
                dx = m / gp
            else:
                dx = (1 + abx)*np.exp(iter*1j)
            x1 = x - dx
            if x == x1:  # converged
                return x, iter
            if iter % MT != 0:
                x = 1 * x1
            else:  # Every so often we take a fractional step, to break any limit cycle (itself a rare occurrence).
                x = x - FRAC[int(iter / MT) - 1] * dx

    raise LaguerreIterationError('Too many iterations - Try a different starting guess for the root.')
    # Very unusual — can occur only for complex roots.


def zroots(a, polish=True):
    """Driver routine that calls laguerre in succession for each root, performs the deflation,
    optionally polishes the roots by the same Laguerre method, finally sorts the roots.
    a[]    - m+1 coefficients of complex polynomial
    polish - if roots to be improved"""

    EPS = 2.0e-6
    m = len(a)-1  # degree of polynomial based on number of coefficients
    roots = np.empty(m, dtype=np.complex64)  # roots have indexes [0..m-1]

    # Given the degree m and the m+1 complex coefficients a[0, .., m+1] of the polynomial SUM(i=0 to m)a(i)x**i,
    # this routine successively calls laguerre and finds all m complex roots roots[0, ... ,m-1].
    # The logical variable polish should be input as "True" if polishing (also by Laguerre’s method) is desired,
    # "False" if the roots will be subsequently polished by other means.

    ad = a.copy()  # Copy of coefficients for successive deflation.
    for j in range(m, 0, -1):      # Loop over each root to be found
        x = 0 + 0j  # Start at zero to favor convergence to the smallest remaining root.
        x, its = laguerre(ad, j, x)  # Find the root, its - number of iterations
        if fabs(x.imag) <= 2 * EPS * fabs(x.real):
            x = x.real + 0j
        roots[j-1] = x
        b = ad[j]  # Forward deflation.
        for jj in range(j-1, -1, -1):
            c = ad[jj]
            ad[jj] = b
            b = x * b + c
    if polish:
        for j in range(1, m+1):  # Polish the roots using the undeflated coefficients.
            roots[j-1], its = laguerre(a, m, roots[j-1])

    # sort ascending, firs real then imag but! if value close to zero for real sort imag first
    # :todo: for now on values close to zero are zeroed to get sort by imag value. Anyway: what should be order of roots???
    roots.real[abs(roots.real) < 1e-6] = 0.0
    roots.imag[abs(roots.imag) < 1e-6] = 0.0
    return np.sort_complex(roots)


def conjugates(_roots):
    """Roots are sorted primarily for real part but order of conjugates not necessarily will be + - + - + - so conjugates are chosen by sign of imag part"""
    return [_r for _r in _roots if _r.imag >= 0]


if __name__ == '__main__':
    pass
    np.set_printoptions(suppress=True, precision=5)
    roots = []
    for i in range(100):
        r = np.random.random(6) + np.random.random(6) * 1j
        roots.append(r)

    for i in range(100):
        r = np.ones(6) + np.random.random(6) * 1j
        roots.append(r)

    for i in range(100):
        r = np.random.random(6) + np.ones(6) * 1j
        roots.append(r)

    for i in range(100):
        r = np.ones(6) + (np.random.random(6)*0.01) * 1j
        roots.append(r)

    for i in range(100):
        r = (np.random.random(6)*0.001) + np.ones(6) * 1j
        roots.append(r)

    r = np.ones(6) + np.ones(6) * 1j
    roots.append(r)
    r = np.zeros(6) + np.ones(6) * 1j
    roots.append(r)

    e = 0
    d = 0
    for t, r in enumerate(roots):
        r.sort()
        p = np.polynomial.polynomial.polyfromroots(r)
        # rr = zroots(p, polish=True)
        rr = np.polynomial.polynomial.polyroots(p)
        rr.sort()
        print(rr)
        if not np.isclose(r, rr, atol=0.01, rtol=0).all():
            e += 1
    print('tests:  ', t+1)
    print('errors: ', e)
