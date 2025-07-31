"""
Distribution classes that provide compatibility with scipy.stats and torch.distributions
for better integration with neural network methods while maintaining API compatibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
import numpy as np
import torch
import torch.distributions as torch_dist
from scipy import stats


class Distribution(ABC):
    """
    Base class for all distributions, providing a common interface
    that works with both scipy and torch backends.
    """

    def __init__(self, backend: str = "scipy", rng: np.random.Generator | None = None):
        """
        Parameters
        ----------
        backend : str
            Either "scipy" or "torch" for the computational backend
        rng : np.random.Generator, optional
            Random number generator instance. If None, creates own RNG with random seed.
        """
        if backend not in ["scipy", "torch"]:
            msg = f"Unsupported backend: {backend}. Must be 'scipy' or 'torch'"
            raise ValueError(msg)

        self.backend = backend
        self.rng = rng if rng is not None else np.random.default_rng()

    def draw(self, n: int = 1) -> np.ndarray:
        """Draw n samples from the distribution"""
        if self.backend == "scipy":
            return np.asarray(self._dist.rvs(size=n, random_state=self.rng))
        if self.backend == "torch":
            samples = self._dist.sample((n,))
            return samples.detach().cpu().numpy()
        msg = f"Unsupported backend: {self.backend}"
        raise ValueError(msg)

    @abstractmethod
    def discretize(self, **kwargs) -> DiscreteDistribution:
        """Discretize the distribution"""

    @property
    @abstractmethod
    def mean(self) -> float:
        """Mean of the distribution"""

    @property
    @abstractmethod
    def std(self) -> float:
        """Standard deviation of the distribution"""


class Normal(Distribution):
    """Normal distribution compatible with skagent.distributions.Normal"""

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 1.0,
        backend: str = "scipy",
        rng: np.random.Generator | None = None,
    ):
        super().__init__(backend, rng)
        self.mu = mu
        self.sigma = sigma

        if self.backend == "scipy":
            self._dist = stats.norm(loc=mu, scale=sigma)
        elif self.backend == "torch":
            self._dist = torch_dist.Normal(
                torch.tensor(mu, dtype=torch.float32),
                torch.tensor(sigma, dtype=torch.float32),
            )

    def discretize(
        self,
        n_points: int = 7,
        sigma_range: float = 3.0,
        N: int | None = None,
        **kwargs,
    ) -> DiscreteDistribution:
        """Discretize using Gauss-Hermite quadrature or uniform grid"""
        # Handle alternative parameter naming
        if N is not None:
            n_points = N

        if self.backend == "scipy":
            # Use Gauss-Hermite quadrature for normal distribution
            points, weights = np.polynomial.hermite.hermgauss(n_points)
            points = points * np.sqrt(2) * self.sigma + self.mu
            weights = weights / np.sqrt(np.pi)
        else:
            # Uniform grid approach
            points = np.linspace(
                self.mu - sigma_range * self.sigma,
                self.mu + sigma_range * self.sigma,
                n_points,
            )
            weights = np.ones(n_points) / n_points

        return DiscreteDistribution(points, weights, var_names=["x"])

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def std(self) -> float:
        return self.sigma


class Lognormal(Distribution):
    """Lognormal distribution compatible with skagent.distributions.Lognormal"""

    def __init__(
        self,
        mean: float = 1.0,
        std: float = 1.0,
        backend: str = "scipy",
        rng: np.random.Generator | None = None,
    ):
        super().__init__(backend, rng)
        # Convert mean/std parameterization to mu/sigma for lognormal
        self.mean_param = mean
        self.std_param = std

        # Handle degenerate case where std=0 (point mass at mean)
        if std == 0:
            self.is_degenerate = True
            self.mu = np.log(mean) if mean > 0 else -np.inf
            self.sigma = 0
            self._dist = None  # Will handle in draw() method
        else:
            self.is_degenerate = False
            # For lognormal: if X ~ LogNormal(mu, sigma), then E[X] = exp(mu + sigma^2/2)
            # We need to solve for mu, sigma given mean and std
            var = std**2
            self.mu = np.log(mean**2 / np.sqrt(var + mean**2))
            self.sigma = np.sqrt(np.log(1 + var / mean**2))

            if self.backend == "scipy":
                self._dist = stats.lognorm(s=self.sigma, scale=np.exp(self.mu))
            elif self.backend == "torch":
                self._dist = torch_dist.LogNormal(
                    torch.tensor(self.mu, dtype=torch.float32),
                    torch.tensor(self.sigma, dtype=torch.float32),
                )

    def draw(self, n: int = 1) -> np.ndarray:
        if self.is_degenerate:
            # Return point mass at mean
            return np.full(n, self.mean_param)

        if self.backend == "scipy":
            return np.asarray(self._dist.rvs(size=n, random_state=self.rng))
        if self.backend == "torch":
            samples = self._dist.sample((n,))
            return samples.detach().cpu().numpy()
        msg = f"Unsupported backend: {self.backend}"
        raise ValueError(msg)

    def discretize(
        self, n_points: int = 7, N: int | None = None, **kwargs
    ) -> DiscreteDistribution:
        # Handle alternative parameter naming
        if N is not None:
            n_points = N

        if self.is_degenerate:
            # Point mass distribution
            points = np.array([self.mean_param])
            weights = np.array([1.0])
            return DiscreteDistribution(points, weights, var_names=["x"])

        # Discretize the underlying normal and transform
        normal_points = np.linspace(
            -3 * self.sigma + self.mu, 3 * self.sigma + self.mu, n_points
        )
        points = np.exp(normal_points)
        weights = np.ones(n_points) / n_points
        return DiscreteDistribution(points, weights, var_names=["x"])

    @property
    def mean(self) -> float:
        return self.mean_param

    @property
    def std(self) -> float:
        return self.std_param


class MeanOneLogNormal(Lognormal):
    """Lognormal distribution with mean normalized to 1.0"""

    def __init__(
        self,
        sigma: float = 1.0,
        backend: str = "scipy",
        rng: np.random.Generator | None = None,
    ):
        # For mean-one lognormal: mu = -sigma^2/2
        mu = -(sigma**2) / 2
        mean = 1.0  # By construction
        std = np.sqrt(np.exp(2 * mu + sigma**2) * (np.exp(sigma**2) - 1))
        super().__init__(mean=mean, std=std, backend=backend, rng=rng)
        self.sigma_param = sigma


class Uniform(Distribution):
    """Uniform distribution"""

    def __init__(
        self,
        low: float = 0.0,
        high: float = 1.0,
        backend: str = "scipy",
        rng: np.random.Generator | None = None,
    ):
        super().__init__(backend, rng)
        self.low = low
        self.high = high

        if self.backend == "scipy":
            self._dist = stats.uniform(loc=low, scale=high - low)
        elif self.backend == "torch":
            self._dist = torch_dist.Uniform(
                torch.tensor(low, dtype=torch.float32),
                torch.tensor(high, dtype=torch.float32),
            )

    def discretize(
        self,
        n_points: int = 7,
        N: int | None = None,
        **kwargs,
    ) -> DiscreteDistribution:
        """Discretize using Gauss-Hermite quadrature or uniform grid"""
        # Handle alternative parameter naming
        if N is not None:
            n_points = N

        # Uniform grid approach
        points = np.linspace(
            self.low,
            self.high,
            n_points,
        )
        weights = np.ones(n_points) / n_points

        return DiscreteDistribution(points, weights, var_names=["x"])

    @property
    def mean(self) -> float:
        return (self.low + self.high) / 2

    @property
    def std(self) -> float:
        return (self.high - self.low) / (2 * math.sqrt(3))


class Bernoulli(Distribution):
    """Bernoulli distribution compatible with skagent.distributions.Bernoulli"""

    def __init__(
        self,
        p: float = 0.5,
        backend: str = "scipy",
        rng: np.random.Generator | None = None,
    ):
        super().__init__(backend, rng)
        self.p = p

        if self.backend == "scipy":
            self._dist = stats.bernoulli(p=p)
        elif self.backend == "torch":
            self._dist = torch_dist.Bernoulli(torch.tensor(p, dtype=torch.float32))

    def discretize(self, **kwargs) -> DiscreteDistribution:
        """Bernoulli is already discrete"""
        points = np.array([0, 1])
        weights = np.array([1 - self.p, self.p])
        return DiscreteDistribution(points, weights, var_names=["x"])

    @property
    def mean(self) -> float:
        return self.p

    @property
    def std(self) -> float:
        return np.sqrt(self.p * (1 - self.p))


class DiscreteDistribution:
    """
    A discrete distribution representation for labeled discrete distributions
    """

    def __init__(
        self,
        points: np.ndarray,
        weights: np.ndarray,
        var_names: list[str] | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.points = np.asarray(points)
        self.weights = np.asarray(weights)
        self.var_names = var_names or ["x"]
        self.rng = rng if rng is not None else np.random.default_rng()

        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)

        # Create variables dict for compatibility
        self.variables = {name: i for i, name in enumerate(self.var_names)}

        # Legacy compatibility attributes
        self.pmv = self.weights  # pmv = probability mass vector

    def draw(self, n: int = 1) -> np.ndarray:
        """Draw samples from the discrete distribution"""
        indices = self.rng.choice(len(self.points), size=n, p=self.weights)
        return self.points[indices]

    @property
    def mean(self) -> float:
        return np.sum(self.points * self.weights)

    @property
    def std(self) -> float:
        mean_val = self.mean
        return np.sqrt(np.sum((self.points - mean_val) ** 2 * self.weights))


class DiscreteDistributionLabeled(DiscreteDistribution):
    """
    Labeled discrete distribution with variable names
    """

    @classmethod
    def from_unlabeled(cls, unlabeled_dist, var_names: list[str]):
        """Create labeled distribution from unlabeled one"""
        if hasattr(unlabeled_dist, "points") and hasattr(unlabeled_dist, "weights"):
            return cls(unlabeled_dist.points, unlabeled_dist.weights, var_names)
        if hasattr(unlabeled_dist, "discretize"):
            # This is one of our distribution objects that needs to be discretized first
            discrete_dist = unlabeled_dist.discretize()
            return cls(discrete_dist.points, discrete_dist.weights, var_names)
        # Handle case where unlabeled_dist is a simple discrete dist
        points = getattr(unlabeled_dist, "xk", None)
        weights = getattr(unlabeled_dist, "pk", None)
        if points is None or weights is None:
            msg = f"Cannot extract points/weights from {type(unlabeled_dist)}"
            raise ValueError(msg)
        return cls(points, weights, var_names)


class IndexDistribution:
    """
    Distribution that varies by index (like age), compatible with skagent.distributions.IndexDistribution
    """

    def __init__(
        self,
        dist_class,
        params_dict: dict[str, list],
        rng: np.random.Generator | None = None,
    ):
        self.dist_class = dist_class
        self.params_dict = params_dict
        self.rng = rng if rng is not None else np.random.default_rng()

        # Create distributions for each index
        self.distributions = []
        n_indices = len(next(iter(params_dict.values())))

        for i in range(n_indices):
            params = {key: values[i] for key, values in params_dict.items()}
            # Pass the same RNG instance to all child distributions
            self.distributions.append(dist_class(**params, rng=self.rng))

    def draw(self, conditions: np.ndarray) -> np.ndarray:
        """Draw samples based on conditions (typically ages)"""
        results = np.zeros_like(conditions, dtype=float)
        for i, condition in enumerate(conditions):
            if condition < len(self.distributions):
                results[i] = self.distributions[condition].draw(1)[0]
            else:
                # Use last distribution if condition exceeds available distributions
                results[i] = self.distributions[-1].draw(1)[0]
        return results


class TimeVaryingDiscreteDistribution:
    """
    Time-varying discrete distribution for compatibility
    """

    def __init__(self, distributions: list[DiscreteDistribution]):
        self.distributions = distributions

    def draw(self, conditions: np.ndarray) -> np.ndarray:
        """Draw samples based on time conditions"""
        results = np.zeros_like(conditions, dtype=float)
        for i, condition in enumerate(conditions):
            if condition < len(self.distributions):
                results[i] = self.distributions[condition].draw(1)[0]
            else:
                results[i] = self.distributions[-1].draw(1)[0]
        return results


def combine_indep_dstns(*distributions) -> DiscreteDistribution:
    """
    Combine independent discrete distributions into a joint distribution
    Compatible with skagent.distributions.combine_indep_dstns
    """
    if len(distributions) == 1:
        return distributions[0]

    # Start with first distribution
    combined_points = distributions[0].points.reshape(-1, 1)
    combined_weights = distributions[0].weights
    var_names = distributions[0].var_names.copy()

    # Combine with each subsequent distribution
    for dist in distributions[1:]:
        # Create meshgrid for all combinations
        new_points = []
        new_weights = []

        for _i, (old_point, old_weight) in enumerate(
            zip(combined_points, combined_weights)
        ):
            for _j, (new_point, new_weight) in enumerate(
                zip(dist.points, dist.weights)
            ):
                combined_point = np.concatenate([old_point.flatten(), [new_point]])
                new_points.append(combined_point)
                new_weights.append(old_weight * new_weight)

        combined_points = np.array(new_points)
        combined_weights = np.array(new_weights)
        var_names.extend(dist.var_names)

    return DiscreteDistribution(combined_points, combined_weights, var_names)


def expected(func, dist: DiscreteDistribution) -> float:
    """
    Compute expected value of a function over a discrete distribution
    Compatible with skagent.distributions.expected
    """
    if hasattr(dist, "points") and hasattr(dist, "weights"):
        # Handle case where points might be multidimensional
        if dist.points.ndim > 1:
            results = []
            for point, weight in zip(dist.points, dist.weights):
                # Create dict mapping variable names to values
                point_dict = {}
                if len(dist.var_names) == len(point):
                    for var_name, value in zip(dist.var_names, point):
                        point_dict[var_name] = value
                else:
                    # Fallback if var_names don't match
                    point_dict = {f"var_{i}": val for i, val in enumerate(point)}

                result = func(point_dict)
                results.append(result * weight)
            return np.sum(results)
        # 1D case - try both scalar and dict-like approaches for compatibility
        results = []
        for point, weight in zip(dist.points, dist.weights):
            try:
                # First try passing scalar value directly (most common case)
                result = func(point)
            except (TypeError, IndexError):
                # If that fails, try with dict-like structure for legacy compatibility
                if len(dist.var_names) == 1:

                    class PointDict:
                        def __init__(self, value, var_name):
                            self.value = value
                            self.var_name = var_name

                        def __getitem__(self, key):
                            if key == self.var_name:
                                return self.value
                            msg = f"Unknown variable: {key}"
                            raise KeyError(msg)

                    point_obj = PointDict(point, dist.var_names[0])
                    result = func(point_obj)
                else:
                    result = func(point)
            results.append(result * weight)
        return np.sum(results)
    msg = "Distribution must have points and weights attributes"
    raise ValueError(msg)
