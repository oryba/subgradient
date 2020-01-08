import numpy as np

from alg import RAlg


def sample(x):
    n = 100
    temp = np.arange(0, n).transpose()
    w = 1.2 ** temp
    while True:
        temp = x - np.ones(len(x))
        f = sum(abs(w * temp))
        g = w * np.sign(temp)
        yield f, g


def test_iterations():
    n = 100
    x = np.zeros(n)
    alpha = 4.0
    h0 = 10.0
    q1 = 1.0
    epsx = 1e-48
    epsg = 1e-42
    maxitn = 500000
    intp = 50
    alg = RAlg()
    xr, fr, itn, nfg, istop = alg.run_b5(lambda arg: next(sample(arg)), x, alpha, h0, q1, epsg, epsx, maxitn, intp)

    dx = np.linalg.norm(xr - np.ones(n))
    print(f"\ndx={dx}")
    assert dx == 1.2509735681486066e-13


def test_epsx():
    n = 100
    x = np.zeros(n)
    alpha = 4.0
    h0 = 10.0
    q1 = 1.0
    epsx = 1e-8
    epsg = 1e-12
    maxitn = 500000
    intp = 50
    alg = RAlg()
    xr, fr, itn, nfg, istop = alg.run_b5(lambda arg: next(sample(arg)), x, alpha, h0, q1, epsg, epsx, maxitn, intp)

    dx = np.linalg.norm(xr - np.ones(n))
    print(f"\ndx={dx}")
    assert dx == 2.4421630540081938e-08


def test_epsg():
    n = 100
    x = np.zeros(n)
    alpha = 4.0
    h0 = 10.0
    q1 = 1.0
    epsx = 1e-68
    epsg = 1e-12
    maxitn = 500000
    intp = 50
    alg = RAlg()
    xr, fr, itn, nfg, istop = alg.run_b5(lambda arg: next(sample(arg)), x, alpha, h0, q1, epsg, epsx, maxitn, intp)

    dx = np.linalg.norm(xr - np.ones(n))
    print(f"\ndx={dx}")
    assert dx == 1.2509735681486066e-13

