import numpy.typing as npt
import numpy as np
from typing import List

class ObjFun(object):

    """
    Generic objective function super-class
    """

    def __init__(self, fstar: float, a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> None:
        """
        Default initialization function that sets:
        :param fstar: f^* value to be reached (can be -inf)
        :param a: domain lower bound vector
        :param b: domain upper bound vector
        """
        self.fstar = fstar
        self.a = a
        self.b = b

    def get_fstar(self) -> float:
        """
        Returns f^*
        :return: f^* value
        """
        return self.fstar

    def get_bounds(self) -> List[npt.NDArray[np.float64]]:
        """
        Returns domain bounds
        :return: list with lower and upper domain bound
        """
        return [self.a, self.b]

    def generate_point(self, rng: np.random.Generator = None) -> npt.NDArray[np.float64]:
        """
        Random point generator placeholder
        :param rng: numpy random generator instance
        :return: random point from the domain
        """
        raise NotImplementedError("Objective function must implement its own random point generator")

    def get_neighborhood(self, x: npt.NDArray[np.float64]) -> List[npt.NDArray[np.float64]]:
        """
        Solution neighborhood generating function placeholder
        :param x: point
        :return: list of points in the neighborhood of the x
        """
        raise NotImplementedError("Objective function must implement its own neighborhood generator")

    def evaluate(self, x: npt.NDArray[np.float64]) -> np.float64:
        """
        Objective function evaluating function placeholder
        :param x: point
        :return: objective function value
        """
        raise NotImplementedError("Objective function must implement its own evaluation")
