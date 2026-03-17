from objfun import ObjFun
import numpy as np
import numpy.typing as npt


class TSPGrid(ObjFun):

    def __init__(self, par_a: int, par_b: int, norm: int = 2, greediness: float = 0.0) -> None:
        """
        TSP on a regular par_a × par_b grid of cities.
        :param par_a: grid width
        :param par_b: grid height
        :param norm: distance norm (2 = Euclidean)
        :param greediness: initial-point generation strategy in [0, 1].
            0 = pure random, 1 = pure nearest-neighbour greedy.
        """
        n = par_a * par_b

        # city coordinates — vectorised
        a_grid, b_grid = np.meshgrid(np.arange(par_a), np.arange(par_b), indexing='ij')
        grid = np.column_stack([a_grid.ravel(), b_grid.ravel()])

        # pairwise distance matrix — vectorised
        diff = grid[:, np.newaxis, :] - grid[np.newaxis, :, :]   # (n, n, 2)
        dist = np.linalg.norm(diff, ord=norm, axis=2)             # (n, n)

        self.fstar = n + np.mod(n, 2) * (2 ** (1 / norm) - 1)
        self.n = n
        self.grid = grid   # (n, 2) city coordinates, useful for plotting
        self.dist = dist
        self.greediness = greediness
        self.a = np.zeros(n - 1, dtype=np.int64)    # n-1: first city is fixed
        self.b = np.arange(n - 2, -1, -1, dtype=np.int64)

    def generate_point(self, rng: np.random.Generator = None) -> npt.NDArray[np.int64]:
        """
        Generates a random (or greedy) initial solution using the Lehmer encoding.

        At each step k the next city is chosen from the remaining unvisited cities:
          - with probability self.greediness: pick the nearest one (greedy)
          - otherwise: pick uniformly at random

        greediness=0 reproduces the original pure-random behaviour exactly
        (same domain bounds [a[k], b[k]] = [0, n-2-k]).
        """
        if rng is None:
            rng = np.random.default_rng()

        # c mirrors the 'c' array in decode(): city 0 sits permanently at index 0
        # and is never re-selected; remaining cities are c[1:].
        c = list(range(self.n))
        x = np.zeros(self.n - 1, dtype=np.int64)
        current_city = 0

        for k in range(self.n - 1):
            remaining = c[1:]           # available cities at this step
            n_rem = len(remaining)

            if n_rem == 1:
                j = 0
            elif rng.random() < self.greediness:
                j = int(np.argmin(self.dist[current_city, remaining]))
            else:
                j = int(rng.integers(0, n_rem))

            x[k] = j
            current_city = remaining[j]
            del c[j + 1]                # remove chosen city from c (index j+1 in c)

        return x

    def decode(self, x: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """
        Decodes a Lehmer-encoded solution vector into an ordered tour.
        x[k] is the index into the remaining unvisited cities at step k.
        Example: x = [1, 2, 2, 1, 0] -> cx = [0, 2, 4, 5, 3, 1] (for n=6)
        """
        cx = np.zeros(self.n, dtype=np.int64)
        # c[0] = 0 is the pre-chosen first city; remaining cities fill c[1:]
        # Using a Python list for O(1) deletion by index
        c = list(range(self.n))
        for k in range(1, self.n):
            ix = x[k - 1] + 1       # +1 because c[0]=0 is never re-selected
            cx[k] = c[ix]
            del c[ix]
        return cx

    def tour_dist(self, cx: npt.NDArray[np.int64]) -> np.float64:
        """
        Computes total tour length (including the closing edge back to city 0).
        Uses vectorised indexing: dist[cx[i], cx[(i+1) % n]] summed over all i.
        """
        return np.float64(np.sum(self.dist[cx, np.roll(cx, -1)]))

    def evaluate(self, x: npt.NDArray[np.int64]) -> np.float64:
        return self.tour_dist(self.decode(x))

    def get_neighborhood(self, x: npt.NDArray[np.int64], d: int) -> list[npt.NDArray[np.int64]]:
        """
        Returns all solutions reachable by incrementing or decrementing one
        component of x by 1 (diameter-1 neighbourhood in the Lehmer encoding).
        """
        assert d == 1, "TSPGrid supports neighbourhood with distance = 1 only"
        nd = []
        for i, xi in enumerate(x):
            if x[i] > self.a[i]:
                xl = x.copy(); xl[i] -= 1; nd.append(xl)
            if x[i] < self.b[i]:
                xu = x.copy(); xu[i] += 1; nd.append(xu)
        return nd
