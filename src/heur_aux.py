import numpy as np
import numpy.typing as npt
from objfun import ObjFun

def is_integer(a: npt.NDArray) -> bool:
    """
    Tests if `a` is integer
    """
    dt = a.dtype
    return dt == np.int16 or dt == np.int32 or dt == np.int64


class Correction:

    """
    Baseline mutation correction strategy - "sticks" the solution to domain boundaries
    """

    def __init__(self, of: ObjFun) -> None:
        self.of = of

    def correct(self, x: npt.NDArray) -> npt.NDArray:
        return np.minimum(np.maximum(x, self.of.a), self.of.b)


class MirrorCorrection(Correction):
    """
    Mutation correction via mirroring
    """

    def __init__(self, of: ObjFun) -> None:
        Correction.__init__(self, of)

    def correct(self, x: npt.NDArray) -> npt.NDArray:
        n = np.size(x)
        d = self.of.b - self.of.a
        for k in range(n):
            if d[k] == 0:
                x[k] = self.of.a[k]
            else:
                de = np.mod(x[k] - self.of.a[k], 2*d[k])
                de = np.amin([de, 2*d[k] - de])
                x[k] = self.of.a[k] + de
        return x


class ExtensionCorrection(Correction):
    """
    Mutation correction via periodic domain extension
    """

    def __init__(self, of: ObjFun) -> None:
        Correction.__init__(self, of)

    def correct(self, x: npt.NDArray) -> npt.NDArray:
        d = self.of.b - self.of.a
        x = self.of.a + np.mod(x - self.of.a, d + (1 if is_integer(x) else 0))
        return x


class Mutation:

    """
    Generic mutation super-class
    """

    def __init__(self, correction: Correction = None) -> None:
        self.correction = correction


class CauchyMutation(Mutation):

    """
    Cauchy mutation
    """

    def __init__(self, r: float, correction: Correction) -> None:
        Mutation.__init__(self, correction)
        self.r = r

    def mutate(self, x: npt.NDArray, rng: np.random.Generator) -> npt.NDArray:
        n = np.size(x)
        u = rng.uniform(low=0.0, high=1.0, size=n)
        r = self.r
        x_new = x + r * np.tan(np.pi * (u - 1 / 2))
        if is_integer(x):
            x_new = np.array(np.round(x_new), dtype=int)  # optional rounding
        x_new_corrected = self.correction.correct(x_new)
        return x_new_corrected


class BitFlipMutation(Mutation):

    """
    Bit-flip mutation for binary {0,1}^n domains.
    Each bit is independently flipped with probability p.
    If no bits flip, one random bit is forced to flip (guarantees a move).
    """

    def __init__(self, p: float = None) -> None:
        """
        :param p: per-bit flip probability. Defaults to None (resolved to 1/n at mutation time).
        """
        super().__init__(correction=None)
        self.p = p

    def mutate(self, x: npt.NDArray, rng: np.random.Generator) -> npt.NDArray:
        n = len(x)
        p = self.p if self.p is not None else 1.0 / n
        flip_mask = rng.random(n) < p
        if not np.any(flip_mask):
            flip_mask[rng.integers(n)] = True   # guarantee at least one flip
        x_new = x.copy()
        x_new[flip_mask] = 1 - x_new[flip_mask]
        return x_new
