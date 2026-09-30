"""Minimal CMA-ES (Hansen, "The CMA Evolution Strategy: A Tutorial", 2016), ask/tell interface,
with the whole state serialisable to JSON so a search over multi-hour training runs survives
restarts. Minimises. Written here rather than installed because the cluster env has no `cma`.
"""
import json
import numpy as np


class CMAES:
    def __init__(self, mean, sigma, popsize=None, seed=0):
        n = len(mean)
        self.n = n
        self.mean = np.asarray(mean, dtype=np.float64)
        self.sigma = float(sigma)
        self.lam = int(popsize or 4 + int(3 * np.log(n)))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.gen = 0
        self.rng_seed = seed

    def _BD(self):
        C = (self.C + self.C.T) / 2
        D2, B = np.linalg.eigh(C)
        return B, np.sqrt(np.maximum(D2, 1e-20))

    def ask(self):
        rng = np.random.default_rng(self.rng_seed * 100003 + self.gen)
        B, D = self._BD()
        z = rng.standard_normal((self.lam, self.n))
        return self.mean + self.sigma * (z * D) @ B.T

    def tell(self, X, f):
        X = np.asarray(X, dtype=np.float64)
        idx = np.argsort(np.asarray(f, dtype=np.float64))
        xs = X[idx[:self.mu]]
        old = self.mean
        self.mean = self.w @ xs
        y = (self.mean - old) / self.sigma
        B, D = self._BD()
        Cinvsqrt = B @ np.diag(1 / D) @ B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (Cinvsqrt @ y)
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chiN
                < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        artmp = (xs - old) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (artmp.T * self.w) @ artmp)
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.gen += 1

    def to_json(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_json(cls, d):
        o = cls.__new__(cls)
        for k, v in d.items():
            setattr(o, k, np.asarray(v) if isinstance(v, list) else v)
        return o


if __name__ == "__main__":
    # Self-test: noisy sphere and Rosenbrock in 10-D, with a JSON round-trip every generation.
    for name, fn, budget in [("sphere", lambda x: float(np.sum((x - 1.5) ** 2)), 60),
                             ("rosenbrock", lambda x: float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)), 400)]:
        es = CMAES(np.zeros(10), 0.5, seed=1)
        best = np.inf
        for _ in range(budget):
            es = CMAES.from_json(json.loads(json.dumps(es.to_json())))
            X = es.ask(); f = [fn(x) for x in X]; es.tell(X, f); best = min(best, min(f))
        print(f"{name}: best {best:.3g} after {budget} generations (lambda={es.lam}), sigma {es.sigma:.3g}")
