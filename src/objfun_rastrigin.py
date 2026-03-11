from objfun import ObjFun
import numpy as np
import numpy.typing as npt


class Rastrigin(ObjFun):
    """
    Rastrigin function (function 6 from GEATbx)
    f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i)),  A = 10
    Global minimum f* = 0 at x* = (0, ..., 0)
    Domain: [-5.12, 5.12]^n
    Based on http://www.geatbx.com/docu/fcnindex-01.html#P140_6243
    """

    A = 10.0

    def __init__(self, n: int, eps: float = 0.01) -> None:
        self.n = n
        a = -5.12 * np.ones(n, dtype=np.float64)
        b = 5.12 * np.ones(n, dtype=np.float64)
        super().__init__(0 + eps, a, b)

    def generate_point(self, rng: np.random.Generator = None) -> npt.NDArray[np.float64]:
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.a, self.b).astype(np.float64)

    def evaluate(self, x: npt.NDArray[np.float64]) -> np.float64:
        return np.float64(self.A * self.n + np.sum(x ** 2 - self.A * np.cos(2 * np.pi * x)))
