from objfun import ObjFun
import numpy as np
import numpy.typing as npt


class Knapsack(ObjFun):

    """
    0/1 Knapsack problem as binary minimisation on {0,1}^n.

    Maximise  sum(values[i] * x[i])
    subject to  sum(weights[i] * x[i]) ≤ capacity

    Transformed to minimisation with a quadratic penalty for capacity violations:
        f(x) = −sum(values · x) + penalty · max(0, sum(weights · x) − capacity)

    fstar is set to −optimal_value, computed exactly by dynamic programming.
    The DP is O(n · capacity) and runs at construction time; feasible for n ≤ ~500.
    """

    def __init__(self, n: int = 50, capacity_ratio: float = 0.4,
                 seed: int = 42, penalty: float = 100.0,
                 target_pct: float = 1.0) -> None:
        """
        :param n:               number of items
        :param capacity_ratio:  knapsack capacity as a fraction of total weight
        :param seed:            random seed for reproducible instance generation
        :param penalty:         penalty per unit of excess weight
                                Should satisfy  penalty > max(values / weights)
                                to ensure the unconstrained optimum is feasible.
        :param target_pct:      fraction of the DP optimal value that counts as success.
                                1.0 = exact optimum required; 0.95 = 95 % of optimal suffices.
        """
        rng = np.random.default_rng(seed)
        self.weights = rng.integers(1, 21, size=n, dtype=np.int64)
        self.values  = rng.integers(1, 21, size=n, dtype=np.int64)
        self.capacity = int(np.floor(capacity_ratio * float(np.sum(self.weights))))
        self.penalty  = float(penalty)
        self.n_items  = n

        self.optimal_value = self._dp_solve()
        fstar = -float(self.optimal_value) * target_pct

        a = np.zeros(n, dtype=np.int64)
        b = np.ones(n,  dtype=np.int64)
        super().__init__(fstar=fstar, a=a, b=b)

    # ------------------------------------------------------------------
    # DP solver
    # ------------------------------------------------------------------
    def _dp_solve(self) -> int:
        """Standard 0/1 knapsack DP.  Returns the optimal total value."""
        W  = self.capacity
        dp = np.zeros(W + 1, dtype=np.int64)
        for i in range(self.n_items):
            wi = int(self.weights[i])
            vi = int(self.values[i])
            for w in range(W, wi - 1, -1):
                if dp[w - wi] + vi > dp[w]:
                    dp[w] = dp[w - wi] + vi
        return int(dp[W])

    # ------------------------------------------------------------------
    # ObjFun interface
    # ------------------------------------------------------------------
    def generate_point(self, rng: np.random.Generator = None) -> npt.NDArray[np.int64]:
        if rng is None:
            rng = np.random.default_rng()
        return rng.integers(0, 2, size=self.n_items, dtype=np.int64)

    def get_neighborhood(self, x: npt.NDArray[np.int64], d: int = 1) -> list:
        """Hamming-1 neighbourhood: all single bit-flips."""
        assert d == 1, "Knapsack supports neighbourhood distance = 1 only"
        nd = []
        for i in range(self.n_items):
            xn = x.copy()
            xn[i] = 1 - xn[i]
            nd.append(xn)
        return nd

    def evaluate(self, x: npt.NDArray[np.int64]) -> float:
        total_value  = float(np.dot(self.values,  x))
        total_weight = float(np.dot(self.weights, x))
        excess = max(0.0, total_weight - self.capacity)
        return -total_value + self.penalty * excess

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def solution_info(self, x: npt.NDArray[np.int64]) -> dict:
        """Return a human-readable summary dict for solution x."""
        total_value  = int(np.dot(self.values,  x))
        total_weight = int(np.dot(self.weights, x))
        return {
            'value':    total_value,
            'weight':   total_weight,
            'capacity': self.capacity,
            'feasible': total_weight <= self.capacity,
            'n_items':  int(np.sum(x)),
        }
