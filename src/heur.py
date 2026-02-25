import numpy as np
import pandas as pd
import numpy.typing as npt
from objfun import ObjFun

class StopCriterion(Exception):

    """
    Dedicated exception class, to make stopping of the heuristic more transparent
    """

    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return repr(self.value)


class Heuristic:

    """
    Generic heuristic super-class
    """

    def __init__(self, of: ObjFun, maxeval: int) -> None:
        """
        Initialization function
        :param of: any objective function to be optimized
        :param maxeval: maximum allowed number of evaluations
        """
        self.of = of
        self.maxeval = maxeval
        self.best_y = np.inf
        self.best_x = None
        self.neval = 0

    def evaluate(self, x: npt.NDArray[np.float64]) -> np.float64:
        """
        Single evaluation of the objective function
        :param x: point to be evaluated
        :return: corresponding objective function value
        """
        y = self.of.evaluate(x)
        self.neval += 1
        if y < self.best_y:
            self.best_y = y
            self.best_x = x
        if y <= self.of.get_fstar():
            raise StopCriterion('Found solution with desired fstar value')
        if self.neval == self.maxeval:
            raise StopCriterion('Exhausted maximum allowed number of evaluations')
        return y

    def report_end(self) -> dict:
        """
        Returns report after heuristic has finished
        :return: dict with all necessary data
        """
        return {
            'best_y': self.best_y,
            'best_x': self.best_x,
            'neval': self.neval if self.best_y <= self.of.get_fstar() else np.inf
        }

    def search(self) -> None:
        """
        Core searching function placeholder
        :return: end result report
        """
        raise NotImplementedError("Heuristic must implement its own search function")
