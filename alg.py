import typing
from textwrap import dedent

import numpy as np


class RAlg:
    def __init__(self, silent=False):
        self.silent = silent

    def log(self, iteration, best, calls):
        print(f"{iteration}\t{best}\t{calls}")

    def header(self, alpha, h0, q1, epsg, epsx, maxitn, intp):
        if self.silent:
            return
        print(dedent(
            f"""
                Initial step size: {h0}
                Step decrease coefficient: {q1}
                Space dilation coefficient: {alpha}
                Stop parameters:
                 - epsx: {epsx}
                 - epsg: {epsg}
                 - max iteration number: {maxitn}
                Output interval: {intp}

                Iteration\tBest\tCalls"""
        ))

    def run_b5(
            self,
            func: typing.Callable[[np.ndarray], tuple],
            x: np.ndarray,
            alpha: float,
            h0: float,
            q1: float,
            epsg: float,
            epsx: float,
            maxitn: int,
            intp: int
    ):
        itn = 0
        B = np.eye(len(x))
        hs = h0
        lsa = 0
        lsm = 0
        xr = x.copy()
        fr, g0 = func(xr)
        nfg = 1
        self.header(alpha, h0, q1, epsg, epsx, maxitn, intp)
        if np.linalg.norm(g0) < epsg:
            istop = 2
            return self._result(xr, fr, itn, nfg, istop)

        for itn in range(1, maxitn):
            g1 = B.transpose() @ g0
            dx = B @ g1 / np.linalg.norm(g1)
            d = 1
            ls = 0
            ddx = 0
            while d > 0:
                x -= hs * dx
                ddx += hs * np.linalg.norm(dx)
                f, g1 = func(x)
                nfg += 1
                if f < fr:
                    fr = f
                    xr = x.copy()
                if np.linalg.norm(g1) < epsg:
                    istop = 2
                    return self._result(xr, fr, itn, nfg, istop)

                ls += 1
                if ls > 500:
                    istop = 5
                    return self._result(xr, fr, itn, nfg, istop)

                d = dx.transpose() @ g1

            if ls == 1:
                hs *= q1

            lsa = lsa + ls
            lsm = max(lsm, ls)

            if itn % intp == 0:
                self.log(itn, fr, nfg)
                lsa = 0
                lsm = 0

            if ddx < epsx:
                istop = 3
                return self._result(xr, fr, itn, nfg, istop)

            dg = B.transpose() @ (g1 - g0)
            xi = dg / np.linalg.norm(dg)
            B += ((1 / alpha - 1) * B @ xi * xi.reshape(-1, 1)).transpose()

            g0 = g1.copy()
        istop = 4
        return self._result(xr, fr, itn, nfg, istop)

    def _result(self, xr, fr, itn, nfg, istop):
        self.log(itn, fr, nfg)
        return xr, fr, itn, nfg, istop
