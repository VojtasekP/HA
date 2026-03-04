from heur import Heuristic, StopCriterion
from objfun import ObjFun
from heur_aux import Mutation
import numpy as np
import pandas as pd
from typing import Any, Dict


class FastSimulatedAnnealing(Heuristic):

    """
    Implementation of Fast Simulated Annealing heuristic
    """

    def __init__(self, of: ObjFun, maxeval: int, T0: float, n0: int, alpha: float,
                 mutation: Mutation, seed: int = None) -> None:
        """
        Initialization
        :param of: any objective function to be optimized
        :param maxeval: maximum allowed number of evaluations
        :param T0: initial temperature
        :param n0: cooling strategy parameter - number of steps
        :param alpha: cooling strategy parameter - exponent
        :param mutation: mutation to be used for the specific objective function (see heur_aux.py)
        :param seed: random seed for reproducibility (None = non-deterministic)
        """
        Heuristic.__init__(self, of, maxeval, seed)

        self.T0 = T0
        self.n0 = n0
        self.alpha = alpha
        self.mutation = mutation
        self._log_data = []

    def log(self, data: dict) -> None:
        self._log_data.append(data)

    def report_end(self) -> Dict[str, Any]:
        result = Heuristic.report_end(self)
        result['log_data'] = pd.DataFrame(self._log_data)
        return result

    def search(self) -> Dict[str, Any]:
        """
        Core searching function
        :return: end result report
        """
        try:
            x = self.of.generate_point(self.rng)
            f_x = self.evaluate(x)
            while True:
                k = self.neval - 1  # because of the first obj. fun. evaluation
                t0 = self.T0
                n0 = self.n0
                alpha = self.alpha
                t = t0 / (1 + (k / n0) ** alpha) if alpha > 0 else t0 * np.exp(-(k / n0) ** -alpha)

                y = self.mutation.mutate(x, self.rng)
                f_y = self.evaluate(y)
                s = (f_x - f_y) / t
                swap = self.rng.uniform() < 1/2 + np.arctan(s)/np.pi
                self.log({'step': k, 'x': x, 'f_x': f_x, 'y': y, 'f_y': f_y, 'T': t, 'swap': swap})
                if swap:
                    x = y
                    f_x = f_y

        except StopCriterion:
            return self.report_end()
