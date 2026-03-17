from objfun import ObjFun
import numpy as np
import numpy.typing as npt


class HMeans(ObjFun):
    """
    Heuristic clustering inspired by k-means
    """

    def __init__(self, dim: int = 2, n_clu: int = 3, n_pts: int = 15, sigma: float = 0.05,
                 eps: float = 0.1, lambda_: float = 10.0, seed: int = 10) -> None:
        """
        Pseudo-randomly generates cluster centroids and data point matrix
        :param dim: dimension
        :param n_clu: number of clusters
        :param n_pts: number of data points per cluster
        :param sigma: std dev of points around each cluster centre (larger = more overlap)
        :param eps: tolerance from f*
        :param lambda_: penalization factor for solutions using fewer clusters than n_clu
        :param seed: RNG seed for data generation
        """
        rng = np.random.default_rng(seed)
        C = rng.uniform(size=(n_clu, dim))  # cluster centers

        # generate points for each cluster
        X = np.concatenate([rng.normal(c, sigma, size=(n_pts, dim)) for c in C], axis=0)

        self.n_clu = n_clu
        self.dim = dim
        self.lambda_ = lambda_  # store penalization factor
        a = X.min(axis=0)
        self.a = np.concatenate([a for i in range(n_clu)])  # repeat for each solution centroid
        b = X.max(axis=0)
        self.b = np.concatenate([b for i in range(n_clu)])  # -- // --
        self.X = X
        self.C = C
        x_centroids = self.encode_solution(C)
        self.fstar = self.evaluate(x_centroids) + eps

    def encode_solution(self, C: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Encodes centroids into solution usable by heuristics
        :param C: matrix with centroids
        :return: array
        """
        return C.flatten()

    def decode_solution(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Decodes heuristic solution into centroids
        :param x: array
        :return: matrix with centroids
        """
        return np.reshape(x, (self.n_clu, self.dim))

    def generate_point(self, rng: np.random.Generator = None) -> npt.NDArray[np.float64]:
        """
        Generates a random solution point
        :return: array of shape (n_clu * dim,)
        """
        if rng is None:
            rng = np.random.default_rng()
        C = rng.uniform(size=(self.n_clu, self.dim))  # randomly generates centroids
        return self.encode_solution(C)

    def evaluate(self, x: npt.NDArray[np.float64]) -> np.float64:
        """
        Computes sum of squares of distances from data points to their nearest centroids
        with penalization for solutions using fewer clusters than n_clu
        """
        C = self.decode_solution(x)

        # For each data point, compute distances to all centroids: shape (n_pts*n_clu, n_clu)
        labels = np.zeros(self.X.shape[0], dtype=int)
        ssq = 0.0
        for i, pt in enumerate(self.X):
            d = np.linalg.norm(C - pt, axis=1)
            ix = d.argmin()
            ssq += d[ix] ** 2
            labels[i] = ix

        penalty = self.lambda_ * (self.n_clu - len(np.unique(labels)))
        return ssq + penalty

    def get_cluster_labels(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """
        Returns array with cluster labels [0; n_clu] for a given solution vector
        """
        C = self.decode_solution(x)
        labels = np.zeros(self.X.shape[0], dtype=int)
        for i, pt in enumerate(self.X):
            d = np.linalg.norm(C - pt, axis=1)
            labels[i] = d.argmin()
        return labels
