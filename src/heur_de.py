from heur import Heuristic, StopCriterion
from objfun import ObjFun
import numpy as np
import numpy.typing as npt
from typing import Any, Dict


class DifferentialEvolution(Heuristic):
    """
    Differential Evolution heuristic supporting three mutation strategies:

      DE/rand/1:             y = a + F*(b - c)
      DE/best/1:             y = best + F*(b - c)
      DE/current-to-best/1: y = x + F*(best - x) + F*(b - c)

    References:
      [1] https://en.wikipedia.org/wiki/Differential_evolution
      [2] https://stackoverflow.com/questions/20393102/all-versions-of-differential-evolution-algorithm
    """

    MUTATIONS = ('rand/1', 'best/1', 'current-to-best/1')

    def __init__(self, of: ObjFun, maxeval: int, N: int, CR: float, F: float,
                 mutation: str = 'rand/1', seed: int = None) -> None:
        """
        :param of: objective function
        :param maxeval: maximum number of evaluations
        :param N: population size (≥4 for rand/1, ≥3 for best/1 and current-to-best/1)
        :param CR: crossover probability in [0, 1]
        :param F: differential weight in [0, 2]
        :param mutation: mutation strategy — one of 'rand/1', 'best/1', 'current-to-best/1'
        :param seed: random seed for reproducibility (None = non-deterministic)
        """
        Heuristic.__init__(self, of, maxeval, seed)
        assert mutation in self.MUTATIONS, \
            f"Unknown mutation strategy '{mutation}'. Choose from: {self.MUTATIONS}"
        min_N = 4 if mutation == 'rand/1' else 3
        assert N >= min_N, f'N must be at least {min_N} for {mutation}'
        self.N = N
        self.n = np.size(of.a)
        assert 0 <= CR <= 1, 'CR must be in [0, 1]'
        self.CR = CR
        assert 0 <= F <= 2, 'F must be in [0, 2]'
        self.F = F
        self.mutation = mutation

    def search(self) -> Dict[str, Any]:
        try:
            # Initialization
            pop_X = np.zeros([self.N, self.n], dtype=self.of.a.dtype)  # population solution vectors
            pop_f = np.zeros(self.N)  # population objective function values
            for i in np.arange(self.N):
                x = self.of.generate_point(self.rng)
                pop_X[i, :] = x
                pop_f[i] = self.evaluate(x)

            # Evolution iteration
            while True:
                best_ix = np.argmin(pop_f)
                for i in range(self.N):
                    x = pop_X[i]
                    R = self.rng.integers(low=0, high=self.n)

                    if self.mutation == 'rand/1':
                        agents = self.rng.choice(np.delete(np.arange(self.N), i), 3, replace=False)
                        a = pop_X[agents[0]]
                        b, c = pop_X[agents[1]], pop_X[agents[2]]
                        donor = a + self.F * (b - c)

                    elif self.mutation == 'best/1':
                        agents = self.rng.choice(np.delete(np.arange(self.N), i), 2, replace=False)
                        b, c = pop_X[agents[0]], pop_X[agents[1]]
                        donor = pop_X[best_ix] + self.F * (b - c)

                    else:  # 'current-to-best/1'
                        agents = self.rng.choice(np.delete(np.arange(self.N), i), 2, replace=False)
                        b, c = pop_X[agents[0]], pop_X[agents[1]]
                        donor = x + self.F * (pop_X[best_ix] - x) + self.F * (b - c)

                    mask = self.rng.random(self.n) < self.CR
                    mask[R] = True  # guarantee at least one dimension is mutated
                    y = np.where(mask, donor, x)
                    y = np.clip(y, self.of.a, self.of.b)
                    f_y = self.evaluate(y)
                    if f_y < pop_f[i]:
                        pop_X[i] = y
                        pop_f[i] = f_y
                        if f_y < pop_f[best_ix]:
                            best_ix = i

        except StopCriterion:
            return self.report_end()
