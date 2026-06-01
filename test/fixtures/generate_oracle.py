#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference-oracle fixture generator for the Julia FastARD port.

Builds deterministic input datasets, fits the Python reference
`RegressionFastARDExtended` (rank-1 algorithm with deletion-priority +
alignment test, identical to bayesSparsify.m), and saves BOTH the raw
inputs and the fitted outputs to disk so the Julia test suite can load
the EXACT same inputs and compare against these reference outputs.

The raw input arrays are saved because the Julia RNG produces different
values than numpy; Julia loads the saved X/y rather than regenerating.
"""
import json
import os
import sys

import numpy as np

# Make the reference implementation importable.
HERE = os.path.dirname(os.path.abspath(__file__))
PLANNING_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "planning_files"))
sys.path.insert(0, PLANNING_DIR)

from fastard_matlab import RegressionFastARDExtended  # noqa: E402


def legendre_polynomials(x, max_degree):
    """Legendre polynomials P0..P_max_degree evaluated at x (in [-1,1]).

    Mirrors the recurrence in examples/ishigami_comparison_test.jl exactly:
        P0 = 1, P1 = x,
        (i) P_i = ((2i-1) x P_{i-1} - (i-1) P_{i-2}) / i
    Returns array of shape (len(x), max_degree+1).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    polys = np.zeros((n, max_degree + 1))
    polys[:, 0] = 1.0
    if max_degree >= 1:
        polys[:, 1] = x
    # Julia index i runs 2..max_degree; column i+1 in 1-based => column i in 0-based.
    for i in range(2, max_degree + 1):
        polys[:, i] = ((2 * i - 1) * x * polys[:, i - 1] - (i - 1) * polys[:, i - 2]) / i
    return polys


def generate_pce_basis(X, max_degree):
    """Tensor-product Legendre PCE basis for 3D input.

    Replicates generate_pce_basis() from the Julia example: inputs are
    normalized from [-pi, pi] to [-1, 1] by dividing by pi, univariate
    Legendre polynomials are built per dimension, and tensor-product terms
    with total degree i+j+k <= max_degree are stacked in the SAME nested
    loop order as Julia (i outer, j middle, k inner).
    """
    n_samples, n_dims = X.shape
    assert n_dims == 3, "Expected 3D input for Ishigami function"
    X_normalized = X / np.pi
    polys = [legendre_polynomials(X_normalized[:, d], max_degree) for d in range(n_dims)]

    basis_terms = []
    for i in range(0, max_degree + 1):
        for j in range(0, max_degree + 1):
            for k in range(0, max_degree + 1):
                if i + j + k <= max_degree:
                    term = polys[0][:, i] * polys[1][:, j] * polys[2][:, k]
                    basis_terms.append(term)
    return np.column_stack(basis_terms)


def ishigami(X, a=7.0, b=0.1):
    """Ishigami: f(x) = sin(x1) + a*sin(x2)^2 + b*x3^4*sin(x1)."""
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return np.sin(x1) + a * np.sin(x2) ** 2 + b * x3 ** 4 * np.sin(x1)


def build_datasets():
    """Construct all four input datasets deterministically."""
    datasets = {}

    # (a) multicollinear -- hand-specified, no RNG.
    X_mc = np.array(
        [
            [0.1, -0.1, -0.2, 0.02],
            [0.3, -0.3, -0.6, 0.06],
            [0.4, -0.4, -0.8, 0.08],
            [0.5, -0.5, -1.0, 0.1],
        ],
        dtype=np.float64,
    )
    y_mc = np.array([2.0, 6.0, 8.0, 10.0], dtype=np.float64)
    datasets["multicollinear"] = (X_mc, y_mc)

    # (b) sparse -- fixed seed, true coef sparse.
    np.random.seed(123)
    X_sp = np.random.randn(50, 20)
    true_coef_sp = np.zeros(20)
    true_coef_sp[[0, 4, 9]] = [2.0, -1.5, 3.0]
    y_sp = X_sp @ true_coef_sp + 0.1 * np.random.randn(50)
    datasets["sparse"] = (X_sp, y_sp)

    # (c) wellcond -- fixed seed, well-conditioned.
    np.random.seed(123)
    n, p = 30, 10
    X_wc = np.random.randn(n, p)
    true_coef_wc = np.random.randn(p)
    y_wc = X_wc @ true_coef_wc + 0.01 * np.random.randn(n)
    datasets["wellcond"] = (X_wc, y_wc)

    # (d) ishigami -- Legendre-PCE basis, seed reset right before generation.
    np.random.seed(123)
    X_train = 2 * np.pi * np.random.rand(300, 3) - np.pi
    f = ishigami(X_train)
    noise_std = 0.5
    y_ish = f + noise_std * np.random.randn(300)
    Psi = generate_pce_basis(X_train, max_degree=4)
    datasets["ishigami"] = (Psi, y_ish)

    return datasets


def fit_oracle(X, y):
    """Fit the reference model with the requested settings."""
    model = RegressionFastARDExtended(
        n_iter=1000, compute_score=False, fit_intercept=True
    )
    model.fit(X, y)
    return model


def rmse(model, X, y):
    y_hat = model.predict(X, return_std=False)
    return float(np.sqrt(np.mean((y_hat - y) ** 2)))


def main():
    datasets = build_datasets()

    npz_data = {}
    json_data = {}
    summary_lines = []

    for name, (X, y) in datasets.items():
        model = fit_oracle(X, y)
        coef = np.asarray(model.coef_, dtype=np.float64)
        active = np.asarray(model.active_, dtype=bool)
        alpha = np.asarray(model.alpha, dtype=np.float64)
        beta = float(model.beta)
        intercept = float(model.intercept_)
        active_idx = np.where(active)[0].tolist()
        n_active = int(active.sum())
        train_rmse = rmse(model, X, y)

        # NPZ keys (raw arrays for Julia to load + reference outputs).
        npz_data[f"{name}_X"] = X
        npz_data[f"{name}_y"] = y
        npz_data[f"{name}_coef"] = coef
        npz_data[f"{name}_intercept"] = np.array(intercept)
        npz_data[f"{name}_active"] = active
        npz_data[f"{name}_alpha"] = alpha
        npz_data[f"{name}_beta"] = np.array(beta)

        # JSON (human-readable mirror; alpha cast handles inf -> null).
        json_data[name] = {
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "X": X.tolist(),
            "y": y.tolist(),
            "coef": coef.tolist(),
            "intercept": intercept,
            "active": active.tolist(),
            "active_indices": active_idx,
            "n_active": n_active,
            "alpha": [None if not np.isfinite(a) else float(a) for a in alpha],
            "beta": beta,
            "train_rmse": train_rmse,
            "converged": bool(model.converged),
        }

        summary_lines.append(
            f"{name:>14s}: n_active={n_active:3d}  "
            f"active={active_idx}  beta={beta:.6g}  rmse={train_rmse:.6g}"
        )

    npz_path = os.path.join(HERE, "oracle.npz")
    json_path = os.path.join(HERE, "oracle.json")
    np.savez(npz_path, **npz_data)
    with open(json_path, "w") as fh:
        json.dump(json_data, fh, indent=2)

    print("=" * 78)
    print("FastARD reference oracle -- summary")
    print("=" * 78)
    for line in summary_lines:
        print(line)
    print("-" * 78)
    print(f"Saved NPZ : {npz_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
