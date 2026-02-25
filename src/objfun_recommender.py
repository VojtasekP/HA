from objfun import ObjFun
import numpy as np
import numpy.typing as npt
from typing import List

# Global optimum location
X_OPT = 37
Y_OPT = 68


def _evaluate(x: int, y: int) -> np.float64:
    """
    Same as in visualize_function.py: global bowl + two sharp local minima (traps).
    Minimization objective.
    """
    # Global bowl (deepest -> true global minimum)
    g = 0.25 * ((x - X_OPT) ** 2 + (y - Y_OPT) ** 2)
    # Two sharp local minima ("traps"): same shape but lifted by offsets
    t1 = 0.30 * ((x - 80) ** 2 + (y - 20) ** 2) + 18.0
    t2 = 0.35 * ((x - 15) ** 2 + (y - 85) ** 2) + 22.0
    return np.float64(min(g, t1, t2))


class Recommender(ObjFun):
    """
    2D discrete objective function: global bowl at (37, 68) plus two local traps.
    Same formula as in visualize_function.py. Domain: integer lattice [0, 99] x [0, 99].
    """

    def __init__(self) -> None:
        a = np.array([0, 0], dtype=np.int64)
        b = np.array([99, 99], dtype=np.int64)
        fstar = _evaluate(X_OPT, Y_OPT)
        super().__init__(fstar, a, b)

    def generate_point(self) -> npt.NDArray[np.int64]:
        """
        Random integer point from the domain.
        :return: random point in [0, 99] x [0, 99] (integer coordinates)
        """
        return np.array(
            [np.random.randint(0, 100), np.random.randint(0, 100)],
            dtype=np.int64,
        )

    def get_neighborhood(
        self, x: npt.NDArray[np.int64], d: int
    ) -> List[npt.NDArray[np.int64]]:
        """
        Solution neighborhood (Chebyshev distance <= d, excluding x).
        :param x: integer point (length-2 array)
        :param d: diameter of the neighbourhood
        :return: list of integer points in the neighborhood of x
        """
        xi, yi = int(x[0]), int(x[1])
        neighbors: List[npt.NDArray[np.int64]] = []
        for dx in range(-d, d + 1):
            for dy in range(-d, d + 1):
                if dx == 0 and dy == 0:
                    continue
                nx = xi + dx
                ny = yi + dy
                if 0 <= nx <= 99 and 0 <= ny <= 99:
                    neighbors.append(np.array([nx, ny], dtype=np.int64))
        return neighbors

    def evaluate(self, x: npt.NDArray[np.int64]) -> np.float64:
        """
        Objective function (minimize). Accepts integer coordinates only.
        :param x: integer point (length-2 array), values in [0, 99]
        :return: objective function value
        """
        ix = int(np.clip(np.round(x[0]), 0, 99))
        iy = int(np.clip(np.round(x[1]), 0, 99))
        return _evaluate(ix, iy)
