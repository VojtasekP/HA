from objfun import ObjFun
import numpy as np
import numpy.typing as npt


def _nash_bid(c_obs: float, N: int, C_max: float) -> float:
    return c_obs + (C_max - c_obs) / N


class AuctionBidding(ObjFun):
    """
    First-price sealed-bid procurement auction.

    The agent learns a piecewise-linear bid function b_x: [0, C_max] -> [0, C_max]
    encoded as x ∈ R^(k+1):
        x[0]       — intercept (base markup at zero signal)
        x[1..k]    — slope of each equal-width segment

    Opponents play the noiseless Nash rule using their own noisy cost signal.

    Objective (minimised internally, displayed as positive profit externally):
        f(x) = -(mean(profit) - risk_lambda * std(profit))
    With risk_lambda=0: pure expected-value maximisation.
    With risk_lambda>0: penalises variance — conservative bidder.

    Each evaluate() call draws fresh random auctions (stochastic oracle).
    fstar is set at (1 - rel_eps) of Nash profit so success = "within rel_eps of Nash".

    Display convention: profit = -best_y  (always positive when doing better than break-even).
    """

    def __init__(
        self,
        k: int = 4,
        N_opps: int = 3,
        sigma_us: float = 0.0,
        sigma_opps: list = None,
        M: int = 2000,
        C_max: float = 500.0,
        seed: int = 42,
        rel_eps: float = 0.02,
        risk_lambda: float = 0.0,
    ) -> None:
        self.k = k
        self.N_opps = N_opps
        self.N = N_opps + 1
        self.sigma_us = sigma_us
        self.sigma_opps = sigma_opps if sigma_opps is not None else [0.0] * N_opps
        self.M = M
        self.C_max = C_max
        self.seed = seed
        self.risk_lambda = risk_lambda
        self._rng = np.random.default_rng(seed)   # seeded once; advances on every evaluate()

        # All dimensions normalised to [0, 1]:
        #   x[0] = intercept / (C_max / N)  → actual intercept = x[0] * C_max/N
        #   x[1..k] = fractional slope (unchanged)
        a = np.zeros(k + 1, dtype=np.float64)
        b = np.ones(k + 1, dtype=np.float64)

        x_nash_vec = self.x_nash
        # nash_profit estimated over many draws so fstar is a stable benchmark
        self.nash_profit = float(np.mean([
            np.mean(self._mc_raw(x_nash_vec)) for _ in range(20)
        ]))
        self.nash_f = -self.nash_profit   # negative (internal minimisation convention)
        fstar = self.nash_f * (1.0 - rel_eps)

        super().__init__(fstar=fstar, a=a, b=b)

    @property
    def x_nash(self) -> npt.NDArray[np.float64]:
        x = np.empty(self.k + 1, dtype=np.float64)
        x[0] = 1.0            # normalised: actual intercept = 1.0 × C_max/N
        x[1:] = (self.N - 1) / self.N
        return x

    def _decode(self, x: npt.NDArray[np.float64], c_obs: float) -> float:
        seg_width = self.C_max / self.k
        b_vals = np.empty(self.k + 1)
        b_vals[0] = x[0] * (self.C_max / self.N)   # un-normalise intercept
        for i in range(1, self.k + 1):
            b_vals[i] = b_vals[i - 1] + x[i] * seg_width

        seg = min(int(c_obs / seg_width), self.k - 1)
        c_lo = seg * seg_width
        bid = b_vals[seg] + x[seg + 1] * (c_obs - c_lo)
        return float(np.clip(bid, c_obs, self.C_max))

    def _decode_batch(self, x: npt.NDArray[np.float64], c_obs_arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Vectorised decode for an array of c_obs values."""
        seg_width = self.C_max / self.k
        b_vals = np.empty(self.k + 1)
        b_vals[0] = x[0] * (self.C_max / self.N)   # un-normalise intercept
        for i in range(1, self.k + 1):
            b_vals[i] = b_vals[i - 1] + x[i] * seg_width
        segs = np.minimum((c_obs_arr / seg_width).astype(int), self.k - 1)
        c_lo = segs * seg_width
        bids = b_vals[segs] + x[segs + 1] * (c_obs_arr - c_lo)
        return np.clip(bids, c_obs_arr, self.C_max)

    def _mc_raw(self, x: npt.NDArray[np.float64],
                rng: np.random.Generator = None, M: int = None) -> npt.NDArray[np.float64]:
        """Returns per-auction profit array (length M). Positive = profit, negative = loss on a win.

        Defaults to the search RNG (self._rng) and self.M, so the oracle stays stochastic
        (it advances on every evaluate()). Pass a fresh rng and a larger M to draw an
        independent, low-variance estimate without disturbing the search — see clean_eval().
        """
        if rng is None:
            rng = self._rng
        if M is None:
            M = self.M
        c_true = rng.uniform(0.0, self.C_max, M)
        c_obs_us = np.clip(c_true + rng.normal(0.0, self.sigma_us, M), 0.0, self.C_max)
        our_bids = self._decode_batch(x, c_obs_us)
        min_opp_bid = np.full(M, np.inf)
        for s in self.sigma_opps:
            c_obs_opp = np.clip(c_true + rng.normal(0.0, s, M), 0.0, self.C_max)
            opp_bids = c_obs_opp + (self.C_max - c_obs_opp) / self.N
            min_opp_bid = np.minimum(min_opp_bid, opp_bids)
        return np.where(our_bids <= min_opp_bid, our_bids - c_true, 0.0)

    def clean_eval(self, x: npt.NDArray[np.float64], M_eval: int = 100000, seed: int = 12345) -> float:
        """Unbiased profit of bid vector x via a fresh, large-M Monte Carlo draw.

        The heuristics minimise a *noisy* oracle, so the best_y they report is the minimum
        over many noisy estimates and is therefore optimistically biased (and their success
        flag can be tripped by a single lucky draw). Re-evaluating best_x here with many more
        auctions removes that selection bias and gives the true expected profit.

        Uses an independent, fixed-seed RNG so (a) the search oracle self._rng is untouched and
        (b) every candidate is scored against the *same* auctions (common random numbers),
        making cross-run/cross-algorithm comparisons reproducible and low-variance.
        """
        rng = np.random.default_rng(seed)
        return float(np.mean(self._mc_raw(x, rng=rng, M=M_eval)))

    def _mc(self, x: npt.NDArray[np.float64]) -> np.float64:
        profits = self._mc_raw(x)
        score = np.mean(profits) - self.risk_lambda * np.std(profits)
        return np.float64(-score)

    def evaluate(self, x: npt.NDArray[np.float64]) -> np.float64:
        return self._mc(x)

    def generate_point(self, rng: np.random.Generator = None) -> npt.NDArray[np.float64]:
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.a, self.b).astype(np.float64)

    def get_neighborhood(self, x: npt.NDArray[np.float64], d: int = 1) -> list:
        step = 0.05 * d
        nd = []
        for i in range(self.k + 1):
            for delta in [-step, step]:
                xn = x.copy()
                xn[i] = np.clip(xn[i] + delta, self.a[i], self.b[i])
                nd.append(xn)
        return nd
