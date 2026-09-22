"""Chernoff sizing, explicit infeasibility, and causal uncertainty debt."""

from dataclasses import dataclass
import math


def success_lower_bound(n, q, k):
    if n < 0 or not 0 <= q <= 1 or k < 0:
        raise ValueError("invalid reliability inputs")
    if k == 0:
        return 1.0
    mu = n * q
    if mu <= k or n < k:
        return 0.0
    return -math.expm1(-((mu - k) ** 2) / (2 * mu))


def required_emissions(k, q, beta):
    if k < 0 or not 0 <= q <= 1 or not 0 < beta < 1:
        raise ValueError("invalid reliability inputs")
    if k == 0:
        return 0
    if q == 0:
        return None
    a = math.log(1 / beta)
    n = math.ceil((k + a + math.sqrt(a * a + 2 * k * a)) / q)
    while success_lower_bound(n, q, k) < 1 - beta - 1e-14:
        n += 1
    return n


def integer_budget(k, q_q15, log_beta_q16):
    if k < 0 or not 0 <= q_q15 <= 32768 or log_beta_q16 <= 0:
        raise ValueError("invalid fixed-point arguments")
    if k == 0:
        return 0
    if q_q15 == 0:
        return None
    a = int(log_beta_q16)
    rad = a * a + 2 * k * a * 65536
    root = math.isqrt(rad)
    root += root * root < rad
    mu = k * 65536 + a + root
    return (mu * 32768 + 65536 * q_q15 - 1) // (65536 * q_q15)


def wilson_lower(successes, attempts, z=1.6448536269514722):
    if not 0 <= successes <= attempts:
        raise ValueError("invalid success counts")
    if not attempts:
        return 0.0
    p = successes / attempts
    return max(
        0.0,
        (
            p
            + z * z / (2 * attempts)
            - z * math.sqrt(p * (1 - p) / attempts + z * z / (4 * attempts**2))
        )
        / (1 + z * z / attempts),
    )


@dataclass
class Decision:
    emissions: int
    target_k: int
    feasible: bool
    bound: float
    reason: str


@dataclass
class SLAController:
    beta: float = 0.05
    energy_cap_uj: float = 1000
    base_energy_uj: float = 120
    packet_energy_uj: float = 20
    max_emissions: int = 64
    max_k: int = 6
    gamma: float = 0.9
    eta: float = 1.0
    debt_threshold: float = 0.1
    kappa_k: float = 1.0
    kappa_beta: float = 2.0
    debt_enabled: bool = True
    debt: float = 0.0

    def decide(self, k_star, q, attempt_rate, remaining_ms, battery=1.0):
        if (
            not 0 <= battery <= 1
            or not 0 <= q <= 1
            or attempt_rate <= 0
            or self.packet_energy_uj <= 0
        ):
            raise ValueError("invalid controller state")
        base = self.max_k if k_star is None else int(k_star)
        excess = max(0.0, self.debt - self.debt_threshold) if self.debt_enabled else 0.0
        target = min(self.max_k, base + math.ceil(self.kappa_k * excess))
        beta = max(1e-9, self.beta * math.exp(-self.kappa_beta * excess))
        desired = required_emissions(target, q, beta)
        limit = min(
            self.max_emissions,
            max(0, math.floor((self.energy_cap_uj - self.base_energy_uj) / self.packet_energy_uj)),
            max(0, math.floor(attempt_rate * max(0, remaining_ms) / 1000 + 1e-12)),
            math.floor(self.max_emissions * (0.25 + 0.75 * battery)),
        )
        n = min(limit, desired if desired is not None else limit)
        feasible = (
            k_star is not None
            and desired is not None
            and desired <= limit
            and self.base_energy_uj <= self.energy_cap_uj
        )
        reason = (
            "ok"
            if feasible
            else ("no_k_meets_validation_target" if k_star is None else "budget_infeasible")
        )
        return Decision(n, target, feasible, success_lower_bound(n, q, target), reason)

    def observe(self, set_size, max_size, miscovered=None):
        violation = self.eta * int(set_size > max_size) + (
            int(bool(miscovered)) if miscovered is not None else 0
        )
        self.debt = (
            self.gamma * self.debt + (1 - self.gamma) * violation if self.debt_enabled else 0.0
        )
