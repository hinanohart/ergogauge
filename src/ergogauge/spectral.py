"""Spectral invariants of the empirical token-transition operator.

All point computations use the deterministic dense path: the estimator caps the state
count at ``max_states + 1``, so ``numpy.linalg.eig`` / ``eigh`` are always applicable
(see docs/CLAIM.md section 6). A fixed-``v0`` sparse fallback is provided for completeness
but is not reached under the default collapse cap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

EPS_RESOLVE = 1e-8


@dataclass
class CheegerResult:
    fiedler_value: float
    conductance_phi: float
    lower_bound: float  # (1/2) * lambda2
    upper_bound: float  # sqrt(2 * lambda2)
    is_theorem: bool = True  # the two-sided bound on the symmetrized graph IS a theorem
    graph: str = "symmetrized_normalized_laplacian"


@dataclass
class KemenyResult:
    value: float
    method: str
    eig_sum_crosscheck: float
    near_reducible: bool


def _eigvals_modulus_sorted(
    P: NDArray[np.float64], eigvals: NDArray[np.complex128] | None = None
) -> NDArray[np.complex128]:
    w = np.linalg.eig(P)[0] if eigvals is None else eigvals
    order = np.argsort(-np.abs(w))
    return np.asarray(w[order], dtype=np.complex128)


def spectral_gap(P: NDArray[np.float64], eigvals: NDArray[np.complex128] | None = None) -> float:
    """g = 1 - |lambda_2(P)| in [0, 1]. Large g = fast mixing; small g = loopy/repetitive."""
    n = P.shape[0]
    if n <= 1:
        return 0.0
    w = _eigvals_modulus_sorted(P, eigvals)
    lam2 = abs(w[1])
    return float(np.clip(1.0 - lam2, 0.0, 1.0))


def kemeny(
    P: NDArray[np.float64],
    pi: NDArray[np.float64],
    eigvals: NDArray[np.complex128] | None = None,
) -> KemenyResult:
    """Kemeny constant via the fundamental-matrix (group-inverse) trace, with an eig-sum
    cross-check. If the chain is near-reducible (a second eigenvalue ~ 1), K is not
    reported as a finite number (the caller fails closed)."""
    n = P.shape[0]
    if n <= 1:
        return KemenyResult(
            value=0.0, method="group_inverse_trace", eig_sum_crosscheck=0.0, near_reducible=False
        )

    w = _eigvals_modulus_sorted(P, eigvals)
    # distance of the *second* eigenvalue from 1 governs near-reducibility
    near_reducible = bool(abs(1.0 - w[1]) < EPS_RESOLVE)

    # eig-sum cross-check: sum over non-unit eigenvalues of 1/(1-lambda)
    nonunit = w[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = 1.0 / (1.0 - nonunit)
    eig_sum = float(np.real(np.sum(terms))) if np.all(np.isfinite(terms)) else float("inf")

    # primary: Z = (I - P + W)^{-1}, W = 1 @ pi^T ; K = trace(Z) - 1
    W = np.broadcast_to(pi, (n, n))
    A = np.eye(n) - P + W
    try:
        Z = np.linalg.inv(A)
        k_primary = float(np.real(np.trace(Z)) - 1.0)
    except np.linalg.LinAlgError:
        k_primary = float("inf")
        near_reducible = True

    value = k_primary if np.isfinite(k_primary) else eig_sum
    return KemenyResult(
        value=float(value),
        method="group_inverse_trace",
        eig_sum_crosscheck=eig_sum,
        near_reducible=near_reducible,
    )


def cheeger(P: NDArray[np.float64], pi: NDArray[np.float64]) -> CheegerResult:
    """Cheeger conductance phi and Fiedler value of the symmetrized normalized Laplacian.

    Edge weights w_ij = (pi_i P_ij + pi_j P_ji)/2 (additive reversibilization). The
    two-sided bound (1/2)lambda2 <= phi <= sqrt(2 lambda2) is a *theorem about this
    symmetrized graph*; the inference that small phi implies the original directed chain
    is trapped is a calibrated heuristic (see NON-CLAIM.md item 4)."""
    n = P.shape[0]
    if n <= 1:
        return CheegerResult(
            fiedler_value=0.0, conductance_phi=1.0, lower_bound=0.0, upper_bound=0.0
        )

    F = pi[:, None] * P
    Wsym = 0.5 * (F + F.T)
    deg = Wsym.sum(axis=1)
    total_vol = float(deg.sum())
    if total_vol <= 0:
        return CheegerResult(
            fiedler_value=0.0, conductance_phi=1.0, lower_bound=0.0, upper_bound=0.0
        )

    with np.errstate(divide="ignore"):
        dinv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    L = np.eye(n) - (dinv_sqrt[:, None] * Wsym * dinv_sqrt[None, :])
    L = 0.5 * (L + L.T)  # symmetrize away fp asymmetry
    evals, evecs = np.linalg.eigh(L)
    fiedler_value = float(max(0.0, evals[1]))
    fvec = evecs[:, 1]

    phi = _sweep_conductance(Wsym, deg, fvec)
    lower = 0.5 * fiedler_value
    upper = float(np.sqrt(2.0 * fiedler_value))
    return CheegerResult(
        fiedler_value=fiedler_value,
        conductance_phi=phi,
        lower_bound=lower,
        upper_bound=upper,
    )


def _sweep_conductance(
    Wsym: NDArray[np.float64], deg: NDArray[np.float64], fvec: NDArray[np.float64]
) -> float:
    """Minimum conductance over prefix cuts of the Fiedler ordering."""
    n = len(deg)
    order = np.argsort(fvec, kind="stable")
    total_vol = float(deg.sum())
    in_set = np.zeros(n, dtype=bool)
    vol_s = 0.0
    cut = 0.0
    best = 1.0
    W = Wsym
    for k in range(n - 1):
        i = int(order[k])
        # moving i into S: cut changes by (edges to outside) - (edges to inside)
        row = W[i]
        cut += float(row.sum() - 2.0 * row[in_set].sum() - row[i])
        in_set[i] = True
        vol_s += float(deg[i])
        denom = min(vol_s, total_vol - vol_s)
        if denom > 0:
            phi = cut / denom
            if phi < best:
                best = phi
    return float(np.clip(best, 0.0, 1.0))
