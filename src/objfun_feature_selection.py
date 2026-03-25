from objfun import ObjFun
import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_wine, load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


class FeatureSelection(ObjFun):

    """
    Feature selection as binary optimisation on {0,1}^n_features.

    A solution x ∈ {0,1}^n selects which features to pass to a k-NN classifier.
    Objective: minimise  1 − mean CV accuracy  (lower is better).
    If no features are selected the solution returns a high penalty value.

    fstar = 1 − target_accuracy  (caller decides what counts as "solved").
    """

    DATASETS = {'wine': load_wine, 'iris': load_iris}

    def __init__(self, dataset: str = 'wine', k: int = 3, cv: int = 5,
                 target_accuracy: float = 0.95, penalty_empty: float = 10.0) -> None:
        """
        :param dataset:          'wine' or 'iris'
        :param k:                number of neighbours for k-NN
        :param cv:               number of cross-validation folds
        :param target_accuracy:  CV accuracy considered "good enough" (defines fstar)
        :param penalty_empty:    objective value returned when no features are selected
        """
        if dataset not in self.DATASETS:
            raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(self.DATASETS)}")

        data = self.DATASETS[dataset]()
        self.X = data.data
        self.y = data.target
        self.feature_names = list(data.feature_names)
        self.n_features = self.X.shape[1]
        self.k = k
        self.cv = cv
        self.penalty_empty = float(penalty_empty)

        a = np.zeros(self.n_features, dtype=np.int64)
        b = np.ones(self.n_features, dtype=np.int64)
        fstar = 1.0 - target_accuracy

        super().__init__(fstar=fstar, a=a, b=b)

    def generate_point(self, rng: np.random.Generator = None) -> npt.NDArray[np.int64]:
        if rng is None:
            rng = np.random.default_rng()
        return rng.integers(0, 2, size=self.n_features, dtype=np.int64)

    def get_neighborhood(self, x: npt.NDArray[np.int64], d: int = 1) -> list:
        """Hamming-1 neighbourhood: all solutions differing from x in exactly one bit."""
        assert d == 1, "FeatureSelection supports neighbourhood distance = 1 only"
        nd = []
        for i in range(self.n_features):
            xn = x.copy()
            xn[i] = 1 - xn[i]
            nd.append(xn)
        return nd

    def evaluate(self, x: npt.NDArray[np.int64]) -> float:
        selected = np.where(x == 1)[0]
        if len(selected) == 0:
            return self.penalty_empty
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=self.k))
        ])
        scores = cross_val_score(pipe, self.X[:, selected], self.y, cv=self.cv)
        return float(1.0 - np.mean(scores))

    def n_selected(self, x: npt.NDArray[np.int64]) -> int:
        return int(np.sum(x))

    def selected_names(self, x: npt.NDArray[np.int64]) -> list:
        return [self.feature_names[i] for i in np.where(x == 1)[0]]
